"""
Dask Distributed Cluster Manager
Connects to external Dask scheduler for batch processing
"""

import asyncio
from typing import Optional, Dict, Any
from loguru import logger

try:
    from dask.distributed import Client, as_completed
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logger.warning("dask[distributed] not installed - Batch processing disabled")

from ..config import ingestion_settings


class DaskClusterManager:
    """
    Manager for Dask distributed cluster
    Handles connection, job submission, and resource management
    """

    def __init__(self):
        self.client: Optional[Client] = None
        self.is_connected = False
        self._jobs_submitted = 0

    async def connect(self):
        """Connect to Dask scheduler"""
        if not DASK_AVAILABLE or not ingestion_settings.dask_enabled:
            logger.info("Dask cluster disabled - skipping connection")
            return

        try:
            # Connect to external Dask scheduler
            self.client = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: Client(
                    ingestion_settings.dask_scheduler_address,
                    timeout='30s',
                    name='neural-oracle-client'
                )
            )

            self.is_connected = True

            # Get cluster info
            cluster_info = self.get_cluster_info()
            logger.info(
                f"Connected to Dask cluster: "
                f"{cluster_info['workers']} workers, "
                f"{cluster_info['cores']} cores, "
                f"{cluster_info['memory']}"
            )

        except Exception as e:
            logger.error(f"Failed to connect to Dask cluster: {e}")
            logger.info("Batch processing will run locally")
            self.is_connected = False

    async def disconnect(self):
        """Disconnect from Dask cluster"""
        if self.client and self.is_connected:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.client.close
                )
                logger.info(f"Dask cluster disconnected. Jobs submitted: {self._jobs_submitted}")
            except Exception as e:
                logger.error(f"Error disconnecting from Dask: {e}")
            finally:
                self.is_connected = False

    def submit_task(self, func: callable, *args, **kwargs):
        """
        Submit a task to the Dask cluster

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future object or result if running locally
        """
        if not self.is_connected:
            # Run locally if cluster not available
            logger.debug("Dask not connected - running task locally")
            return func(*args, **kwargs)

        try:
            future = self.client.submit(func, *args, **kwargs)
            self._jobs_submitted += 1
            return future

        except Exception as e:
            logger.error(f"Error submitting Dask task: {e}")
            # Fallback to local execution
            return func(*args, **kwargs)

    async def submit_async(self, func: callable, *args, **kwargs):
        """Submit task and wait for result asynchronously"""
        future = self.submit_task(func, *args, **kwargs)

        if not self.is_connected:
            return future

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                future.result
            )
            return result
        except Exception as e:
            logger.error(f"Error getting Dask result: {e}")
            return None

    def map_tasks(self, func: callable, items: list):
        """
        Map a function over a list of items in parallel

        Args:
            func: Function to apply
            items: List of items to process

        Returns:
            List of results
        """
        if not self.is_connected:
            # Run locally with list comprehension
            return [func(item) for item in items]

        try:
            futures = self.client.map(func, items)
            results = self.client.gather(futures)
            self._jobs_submitted += len(items)
            return results

        except Exception as e:
            logger.error(f"Error mapping Dask tasks: {e}")
            return [func(item) for item in items]

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the Dask cluster"""
        if not self.is_connected:
            return {
                'connected': False,
                'dask_enabled': ingestion_settings.dask_enabled,
                'dask_available': DASK_AVAILABLE
            }

        try:
            scheduler_info = self.client.scheduler_info()

            return {
                'connected': True,
                'workers': len(scheduler_info['workers']),
                'cores': sum(w['nthreads'] for w in scheduler_info['workers'].values()),
                'memory': sum(w['memory_limit'] for w in scheduler_info['workers'].values()),
                'scheduler': ingestion_settings.dask_scheduler_address,
                'jobs_submitted': self._jobs_submitted
            }

        except Exception as e:
            logger.error(f"Error getting cluster info: {e}")
            return {'error': str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get cluster statistics"""
        return self.get_cluster_info()


# Global cluster manager instance
dask_cluster = DaskClusterManager()
