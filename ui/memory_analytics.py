"""
Memory analytics dashboard UI component.

Provides visualizations and insights for memory patterns.
"""

from nicegui import ui
from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime


class MemoryAnalyticsDashboard:
    """Dashboard component for memory analytics visualization."""
    
    def __init__(self, api_client=None, matter_id: str = None):
        self.api_client = api_client
        self.matter_id = matter_id
        self.dashboard_container = None
        self.analytics_data = None
        self.loading = False
        
    async def load_analytics(self):
        """Load analytics data from API."""
        if not self.api_client or not self.matter_id:
            return
            
        self.loading = True
        try:
            self.analytics_data = await self.api_client.get_memory_analytics(self.matter_id)
        except Exception as e:
            ui.notify(f"Failed to load analytics: {str(e)}", type='error')
            self.analytics_data = None
        finally:
            self.loading = False
    
    def create(self) -> ui.element:
        """Create the analytics dashboard."""
        with ui.card().classes('w-full max-w-6xl') as container:
            self.dashboard_container = container
            
            # Header
            with ui.row().classes('w-full justify-between items-center mb-4'):
                ui.label('Memory Analytics').classes('text-xl font-bold')
                ui.button(
                    icon='refresh',
                    on_click=self._refresh_analytics
                ).props('flat round')
            
            # Loading indicator
            self.loading_spinner = ui.spinner(size='lg').classes('mx-auto')
            self.loading_spinner.visible = False
            
            # Content area
            self.content_area = ui.column().classes('w-full gap-4')
            
            # Load initial data
            asyncio.create_task(self._initial_load())
            
        return container
    
    async def _initial_load(self):
        """Initial load of analytics data."""
        self.loading_spinner.visible = True
        await self.load_analytics()
        self.loading_spinner.visible = False
        self._render_analytics()
    
    async def _refresh_analytics(self):
        """Refresh analytics data."""
        self.loading_spinner.visible = True
        await self.load_analytics()
        self.loading_spinner.visible = False
        self._render_analytics()
    
    def _render_analytics(self):
        """Render analytics visualizations."""
        if not self.analytics_data:
            with self.content_area:
                self.content_area.clear()
                ui.label('No analytics data available').classes('text-gray-500')
            return
        
        with self.content_area:
            self.content_area.clear()
            
            # Error state
            if hasattr(self.analytics_data, 'error') and self.analytics_data.error:
                ui.label(f'Error: {self.analytics_data.error}').classes('text-red-500')
                return
            
            # Summary cards row
            with ui.row().classes('w-full gap-4'):
                self._create_summary_card(
                    'Total Memories',
                    str(self.analytics_data.total_memories),
                    'psychology',
                    'blue'
                )
                
                # Calculate dominant type
                if self.analytics_data.type_distribution:
                    dominant_type = max(
                        self.analytics_data.type_distribution.items(),
                        key=lambda x: x[1]
                    )[0]
                    self._create_summary_card(
                        'Dominant Type',
                        dominant_type,
                        'category',
                        'purple'
                    )
                
                # Calculate top actor
                if self.analytics_data.actor_network:
                    top_actor = max(
                        self.analytics_data.actor_network.items(),
                        key=lambda x: x[1]
                    )[0]
                    self._create_summary_card(
                        'Top Actor',
                        top_actor,
                        'person',
                        'green'
                    )
            
            # Charts row
            with ui.row().classes('w-full gap-4 mt-4'):
                # Type distribution pie chart
                if self.analytics_data.type_distribution:
                    self._create_type_distribution_chart()
                
                # Growth timeline line chart
                if self.analytics_data.growth_timeline:
                    self._create_growth_timeline_chart()
            
            # Actor network bar chart
            if self.analytics_data.actor_network:
                self._create_actor_network_chart()
            
            # Insights section
            if self.analytics_data.insights:
                self._create_insights_section()
            
            # Patterns section
            if self.analytics_data.patterns:
                self._create_patterns_section()
    
    def _create_summary_card(self, title: str, value: str, icon: str, color: str):
        """Create a summary statistic card."""
        with ui.card().classes(f'flex-1 bg-{color}-50'):
            with ui.row().classes('items-center gap-3'):
                ui.icon(icon).classes(f'text-3xl text-{color}-600')
                with ui.column().classes('gap-0'):
                    ui.label(title).classes('text-xs text-gray-600')
                    ui.label(value).classes('text-lg font-bold')
    
    def _create_type_distribution_chart(self):
        """Create type distribution pie chart."""
        with ui.card().classes('flex-1'):
            ui.label('Memory Type Distribution').classes('font-semibold mb-2')
            
            # Prepare data for echart
            data = [
                {'value': count, 'name': mem_type}
                for mem_type, count in self.analytics_data.type_distribution.items()
            ]
            
            # Create pie chart using NiceGUI's echart
            chart_options = {
                'tooltip': {'trigger': 'item'},
                'legend': {
                    'orient': 'vertical',
                    'left': 'left'
                },
                'series': [{
                    'name': 'Memory Type',
                    'type': 'pie',
                    'radius': '50%',
                    'data': data,
                    'emphasis': {
                        'itemStyle': {
                            'shadowBlur': 10,
                            'shadowOffsetX': 0,
                            'shadowColor': 'rgba(0, 0, 0, 0.5)'
                        }
                    }
                }]
            }
            
            ui.echart(chart_options).classes('w-full h-64')
    
    def _create_growth_timeline_chart(self):
        """Create memory growth timeline chart."""
        with ui.card().classes('flex-1'):
            ui.label('Memory Growth Timeline').classes('font-semibold mb-2')
            
            # Prepare data
            months = [item['month'] for item in self.analytics_data.growth_timeline]
            counts = [item['count'] for item in self.analytics_data.growth_timeline]
            
            # Create line chart
            chart_options = {
                'xAxis': {
                    'type': 'category',
                    'data': months
                },
                'yAxis': {
                    'type': 'value',
                    'name': 'Memory Count'
                },
                'series': [{
                    'data': counts,
                    'type': 'line',
                    'smooth': True,
                    'areaStyle': {}
                }],
                'tooltip': {
                    'trigger': 'axis'
                }
            }
            
            ui.echart(chart_options).classes('w-full h-64')
    
    def _create_actor_network_chart(self):
        """Create actor network bar chart."""
        with ui.card().classes('w-full mt-4'):
            ui.label('Top Actors by Mentions').classes('font-semibold mb-2')
            
            # Get top 10 actors
            top_actors = list(self.analytics_data.actor_network.items())[:10]
            
            if top_actors:
                actors = [item[0] for item in top_actors]
                mentions = [item[1] for item in top_actors]
                
                # Create horizontal bar chart
                chart_options = {
                    'xAxis': {
                        'type': 'value',
                        'name': 'Mentions'
                    },
                    'yAxis': {
                        'type': 'category',
                        'data': actors
                    },
                    'series': [{
                        'data': mentions,
                        'type': 'bar',
                        'itemStyle': {
                            'color': '#10b981'
                        }
                    }],
                    'tooltip': {
                        'trigger': 'axis',
                        'axisPointer': {
                            'type': 'shadow'
                        }
                    },
                    'grid': {
                        'left': '20%'
                    }
                }
                
                ui.echart(chart_options).classes('w-full h-80')
    
    def _create_insights_section(self):
        """Create insights section."""
        with ui.card().classes('w-full mt-4'):
            ui.label('Key Insights').classes('font-semibold mb-3')
            
            with ui.column().classes('gap-2'):
                for insight in self.analytics_data.insights:
                    with ui.row().classes('items-center gap-2'):
                        # Choose icon based on insight type
                        icon = self._get_insight_icon(insight.insight)
                        ui.icon(icon).classes('text-blue-600')
                        
                        # Format insight text
                        text = f"{self._format_insight_name(insight.insight)}: "
                        text += f"{insight.interpretation}"
                        
                        if insight.score is not None:
                            text += f" (Score: {insight.score:.2f})"
                        elif insight.rate is not None:
                            text += f" (Rate: {insight.rate:.1f}/period)"
                        elif insight.avg_connections is not None:
                            text += f" (Avg: {insight.avg_connections:.1f} connections)"
                        
                        ui.label(text).classes('text-sm')
    
    def _create_patterns_section(self):
        """Create patterns section."""
        with ui.card().classes('w-full mt-4'):
            ui.label('Identified Patterns').classes('font-semibold mb-3')
            
            with ui.column().classes('gap-3'):
                for pattern in self.analytics_data.patterns:
                    with ui.card().classes('p-3 bg-gray-50'):
                        # Pattern type
                        pattern_name = self._format_pattern_type(pattern.type)
                        ui.label(pattern_name).classes('font-medium text-sm mb-1')
                        
                        # Pattern details
                        if pattern.value:
                            with ui.row().classes('gap-2 items-center'):
                                ui.icon('trending_up').classes('text-sm text-gray-600')
                                text = f"Value: {pattern.value}"
                                if pattern.count:
                                    text += f" ({pattern.count} occurrences)"
                                if pattern.percentage:
                                    text += f" - {pattern.percentage:.1f}%"
                                ui.label(text).classes('text-xs text-gray-700')
                        
                        # Actor patterns
                        if pattern.actors:
                            ui.label('Related Actors:').classes('text-xs text-gray-600 mt-1')
                            with ui.row().classes('gap-2 flex-wrap'):
                                for actor in pattern.actors[:5]:
                                    ui.badge(
                                        f"{actor.get('name', 'Unknown')} ({actor.get('mentions', 0)})"
                                    ).classes('text-xs')
                        
                        # Document patterns
                        if pattern.documents:
                            ui.label('Source Documents:').classes('text-xs text-gray-600 mt-1')
                            with ui.row().classes('gap-2 flex-wrap'):
                                for doc in pattern.documents[:3]:
                                    ui.badge(
                                        f"{doc.get('name', 'Unknown')} ({doc.get('references', 0)} refs)"
                                    ).classes('text-xs')
                        
                        # Temporal patterns
                        if pattern.month:
                            with ui.row().classes('gap-2 items-center mt-1'):
                                ui.icon('calendar_month').classes('text-sm text-gray-600')
                                ui.label(f"Peak month: {pattern.month}").classes('text-xs text-gray-700')
    
    def _get_insight_icon(self, insight_type: str) -> str:
        """Get icon for insight type."""
        icons = {
            'memory_diversity': 'diversity_3',
            'network_complexity': 'hub',
            'growth_rate': 'trending_up',
            'quality_score': 'star',
            'completeness': 'check_circle'
        }
        return icons.get(insight_type, 'insights')
    
    def _format_insight_name(self, insight_type: str) -> str:
        """Format insight type name."""
        return insight_type.replace('_', ' ').title()
    
    def _format_pattern_type(self, pattern_type: str) -> str:
        """Format pattern type name."""
        formatted = pattern_type.replace('_', ' ').title()
        icons = {
            'dominant_memory_type': '🎯',
            'key_actors': '👥',
            'peak_activity': '📈',
            'primary_sources': '📚'
        }
        icon = icons.get(pattern_type, '📊')
        return f"{icon} {formatted}"