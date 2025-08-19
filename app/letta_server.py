"""
Letta Server Management Module

Provides server lifecycle management for the Letta server, including
startup, shutdown, health monitoring, and process management.
"""

import subprocess
import time
import asyncio
import socket
import signal
import os
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import closing
import requests
from threading import Thread, Event

from .logging_conf import get_logger

logger = get_logger(__name__)


class LettaServerManager:
    """
    Manages the Letta server lifecycle.
    
    Supports multiple deployment modes:
    - subprocess: Run server as a Python subprocess (default)
    - docker: Run server in Docker container
    - external: Connect to externally managed server
    """
    
    _instance: Optional['LettaServerManager'] = None
    
    def __new__(cls) -> 'LettaServerManager':
        """Singleton pattern for server manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the server manager."""
        if self._initialized:
            return
        
        # Default configuration
        self.mode = "subprocess"
        self.host = "localhost"
        self.port = 8283
        self.auto_start = True
        self.startup_timeout = 60
        self.health_check_interval = 30
        
        self.process: Optional[subprocess.Popen] = None
        self.docker_container_id: Optional[str] = None
        self._shutdown_event = Event()
        self._health_monitor_thread: Optional[Thread] = None
        self._is_running = False
        
        self._initialized = True
        
        logger.info(
            "LettaServerManager initialized",
            mode=self.mode,
            host=self.host,
            port=self.port,
            auto_start=self.auto_start
        )
    
    def configure(
        self,
        mode: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        auto_start: Optional[bool] = None,
        startup_timeout: Optional[int] = None,
        health_check_interval: Optional[int] = None
    ):
        """
        Configure server manager parameters.
        
        Args:
            mode: Server deployment mode (subprocess, docker, external)
            host: Server host address
            port: Server port
            auto_start: Whether to start server automatically
            startup_timeout: Max seconds to wait for server startup
            health_check_interval: Seconds between health checks
        """
        if mode is not None:
            self.mode = mode
        if host is not None:
            self.host = host
        if port is not None:
            self.port = port
        if auto_start is not None:
            self.auto_start = auto_start
        if startup_timeout is not None:
            self.startup_timeout = startup_timeout
        if health_check_interval is not None:
            self.health_check_interval = health_check_interval
        
        logger.debug(
            "Server manager configured",
            mode=self.mode,
            host=self.host,
            port=self.port
        )
    
    def start(self) -> bool:
        """
        Start the Letta server.
        
        Returns:
            True if server started successfully, False otherwise
        """
        if self._is_running:
            logger.debug("Server already running")
            return True
        
        # Check if a server is already running on the configured port
        if self.health_check():
            logger.info(f"Letta server already running on {self.host}:{self.port}")
            self._is_running = True
            return True
        
        try:
            if self.mode == "subprocess":
                return self._start_subprocess()
            elif self.mode == "docker":
                return self._start_docker()
            elif self.mode == "external":
                return self._check_external()
            else:
                logger.error(f"Unknown server mode: {self.mode}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the Letta server.
        
        Returns:
            True if server stopped successfully, False otherwise
        """
        if not self._is_running:
            logger.debug("Server not running")
            return True
        
        try:
            # Signal health monitor to stop
            self._shutdown_event.set()
            
            if self.mode == "subprocess":
                return self._stop_subprocess()
            elif self.mode == "docker":
                return self._stop_docker()
            elif self.mode == "external":
                # External servers are not managed by us
                self._is_running = False
                return True
                
        except Exception as e:
            logger.error(f"Failed to stop server: {e}")
            return False
    
    def health_check(self) -> bool:
        """
        Check if the server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            # Use the correct health endpoint with trailing slash
            response = requests.get(
                f"http://{self.host}:{self.port}/v1/health/",
                timeout=5,
                allow_redirects=True
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_base_url(self) -> str:
        """Get the base URL for the server."""
        return f"http://{self.host}:{self.port}"
    
    def _find_available_port(self, start_port: int = 8283, max_tries: int = 10) -> int:
        """
        Find an available port starting from start_port.
        
        Args:
            start_port: Port to start searching from
            max_tries: Maximum number of ports to try
            
        Returns:
            Available port number
            
        Raises:
            RuntimeError: If no available port found
        """
        for port in range(start_port, start_port + max_tries):
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                if sock.connect_ex((self.host, port)) != 0:
                    return port
        
        raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_tries}")
    
    def _start_subprocess(self) -> bool:
        """Start server as a subprocess."""
        try:
            # Check if port is available, find alternative if not
            if not self._is_port_available(self.port):
                logger.warning(f"Port {self.port} is in use, finding alternative")
                self.port = self._find_available_port(self.port)
                logger.info(f"Using alternative port: {self.port}")
            
            # Prepare command
            cmd = [
                "letta",
                "server",
                "--host", self.host,
                "--port", str(self.port)
            ]
            
            # Set up environment
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"  # Ensure output is not buffered
            
            # Start server process
            logger.info(f"Starting Letta server subprocess: {' '.join(cmd)}")
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,  # Discard output to avoid blocking
                stderr=subprocess.DEVNULL,  # Discard error output
                env=env,
                start_new_session=True  # Create new process group
            )
            
            # Wait for server to be ready
            if self._wait_for_server():
                self._is_running = True
                self._start_health_monitor()
                logger.info(f"Letta server started successfully on port {self.port}")
                return True
            else:
                logger.error("Server failed to become ready")
                self._cleanup_subprocess()
                return False
                
        except Exception as e:
            logger.error(f"Failed to start subprocess: {e}")
            self._cleanup_subprocess()
            return False
    
    def _stop_subprocess(self) -> bool:
        """Stop the subprocess server."""
        if not self.process:
            return True
        
        try:
            logger.info("Stopping Letta server subprocess")
            
            # Try graceful shutdown first
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                logger.warning("Graceful shutdown failed, forcing kill")
                self.process.kill()
                self.process.wait(timeout=5)
            
            self._cleanup_subprocess()
            self._is_running = False
            logger.info("Letta server stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop subprocess: {e}")
            return False
    
    def _cleanup_subprocess(self):
        """Clean up subprocess resources."""
        if self.process:
            self.process = None
    
    def _start_docker(self) -> bool:
        """Start server in Docker container."""
        try:
            # Check if Docker is available
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                logger.error("Docker not available, falling back to subprocess mode")
                self.mode = "subprocess"
                return self._start_subprocess()
            
            # Check if port is available
            if not self._is_port_available(self.port):
                logger.warning(f"Port {self.port} is in use, finding alternative")
                self.port = self._find_available_port(self.port)
                logger.info(f"Using alternative port: {self.port}")
            
            # Start Docker container
            cmd = [
                "docker", "run", "-d",
                "--name", "letta-server",
                "-p", f"{self.port}:8283",
                "-v", f"{Path.home()}/.letta:/root/.letta",
                "letta/letta:latest",
                "server", "--host", "0.0.0.0", "--port", "8283"
            ]
            
            logger.info("Starting Letta server in Docker container")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.docker_container_id = result.stdout.strip()
                
                # Wait for server to be ready
                if self._wait_for_server():
                    self._is_running = True
                    self._start_health_monitor()
                    logger.info(f"Letta server started in Docker on port {self.port}")
                    return True
                else:
                    self._stop_docker()
                    return False
            else:
                logger.error(f"Failed to start Docker container: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start Docker container: {e}")
            return False
    
    def _stop_docker(self) -> bool:
        """Stop the Docker container."""
        if not self.docker_container_id:
            return True
        
        try:
            logger.info("Stopping Letta Docker container")
            
            # Stop container
            subprocess.run(
                ["docker", "stop", "letta-server"],
                capture_output=True,
                timeout=10
            )
            
            # Remove container
            subprocess.run(
                ["docker", "rm", "letta-server"],
                capture_output=True,
                timeout=5
            )
            
            self.docker_container_id = None
            self._is_running = False
            logger.info("Letta Docker container stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop Docker container: {e}")
            return False
    
    def _check_external(self) -> bool:
        """Check if external server is available."""
        logger.info(f"Checking external Letta server at {self.host}:{self.port}")
        
        if self._wait_for_server(timeout=10):
            self._is_running = True
            self._start_health_monitor()
            logger.info("Connected to external Letta server")
            return True
        else:
            logger.error("External Letta server not available")
            return False
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            return sock.connect_ex((self.host, port)) != 0
    
    def _wait_for_server(self, timeout: Optional[int] = None) -> bool:
        """
        Wait for server to become ready.
        
        Args:
            timeout: Maximum seconds to wait (uses startup_timeout if None)
            
        Returns:
            True if server became ready, False if timeout
        """
        timeout = timeout or self.startup_timeout
        start_time = time.time()
        
        logger.debug(f"Waiting for server to be ready (timeout={timeout}s)")
        
        while time.time() - start_time < timeout:
            if self.health_check():
                logger.debug("Server is ready")
                return True
            
            # Check if subprocess died
            if self.mode == "subprocess" and self.process:
                poll = self.process.poll()
                if poll is not None:
                    logger.error(f"Server process died with code {poll}")
                    return False
            
            time.sleep(0.5)
        
        logger.error(f"Server failed to become ready within {timeout} seconds")
        return False
    
    def _start_health_monitor(self):
        """Start background health monitoring thread."""
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            return
        
        self._shutdown_event.clear()
        self._health_monitor_thread = Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name="LettaHealthMonitor"
        )
        self._health_monitor_thread.start()
        logger.debug("Health monitor started")
    
    def _health_monitor_loop(self):
        """Background loop for health monitoring."""
        consecutive_failures = 0
        max_failures = 3
        
        while not self._shutdown_event.is_set():
            # Wait for interval or shutdown signal
            if self._shutdown_event.wait(self.health_check_interval):
                break
            
            # Perform health check
            if self.health_check():
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                logger.warning(
                    f"Health check failed ({consecutive_failures}/{max_failures})"
                )
                
                # Attempt restart if too many failures
                if consecutive_failures >= max_failures:
                    logger.error("Too many health check failures, attempting restart")
                    if self.mode == "subprocess":
                        self._restart_subprocess()
                    consecutive_failures = 0
        
        logger.debug("Health monitor stopped")
    
    def _restart_subprocess(self):
        """Attempt to restart the subprocess server."""
        try:
            logger.info("Attempting to restart Letta server")
            self._stop_subprocess()
            time.sleep(2)
            self._start_subprocess()
        except Exception as e:
            logger.error(f"Failed to restart server: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        if self.auto_start:
            self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Global server manager instance
server_manager = LettaServerManager()