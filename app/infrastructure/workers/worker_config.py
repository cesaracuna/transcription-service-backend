"""
Worker configuration and startup scripts.
"""

import sys
import logging
from typing import List, Optional

import click
from celery import Celery
from celery.bin import worker

from .celery_app import celery_app
from ...core.config import settings
from ...core.logging import setup_logging, get_logger

logger = get_logger(__name__)


class WorkerManager:
    """Manages Celery worker lifecycle and configuration."""
    
    def __init__(self, celery_app: Celery):
        self.celery_app = celery_app
        self.worker = None
    
    def start_worker(
        self,
        queues: Optional[List[str]] = None,
        concurrency: Optional[int] = None,
        loglevel: str = "INFO"
    ) -> None:
        """
        Start Celery worker with specified configuration.
        
        Args:
            queues: List of queues to consume from
            concurrency: Number of worker processes
            loglevel: Logging level
        """
        logger.info("Starting Celery worker")
        
        # Default queues
        if queues is None:
            queues = ["transcription", "post_processing", "health", "maintenance"]
        
        # Default concurrency (based on CPU cores)
        if concurrency is None:
            import multiprocessing
            concurrency = max(1, multiprocessing.cpu_count() // 2)
        
        # Configure worker
        worker_args = [
            f"--queues={','.join(queues)}",
            f"--concurrency={concurrency}",
            f"--loglevel={loglevel}",
            "--pool=eventlet",  # Use eventlet for I/O bound tasks
            "--without-gossip",  # Reduce network chatter
            "--without-mingle",  # Reduce startup time
            "--without-heartbeat",  # Reduce network overhead
        ]
        
        logger.info(f"Worker configuration: queues={queues}, concurrency={concurrency}")
        
        # Start worker
        self.worker = worker.worker(app=self.celery_app)
        self.worker.run_from_argv("celery", worker_args)
    
    def stop_worker(self) -> None:
        """Stop the Celery worker gracefully."""
        if self.worker:
            logger.info("Stopping Celery worker")
            self.worker.stop()
        else:
            logger.warning("No worker to stop")


def initialize_worker_environment() -> None:
    """Initialize worker environment with proper settings."""
    logger.info("Initializing worker environment")
    
    # Setup logging
    setup_logging(settings.logging, process_name="CeleryWorker")
    
    # Initialize database connection
    from ...infrastructure.database.connection import initialize_database
    initialize_database(settings.database)
    
    # Initialize AI models (lazy loading)
    logger.info("Worker environment initialized")


@click.group()
def cli():
    """Celery worker management CLI."""
    pass


@cli.command()
@click.option('--queues', '-q', default="transcription,post_processing,health,maintenance",
              help='Comma-separated list of queues to consume')
@click.option('--concurrency', '-c', type=int, default=None,
              help='Number of worker processes (default: CPU cores / 2)')
@click.option('--loglevel', '-l', default="INFO",
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              help='Logging level')
def start(queues: str, concurrency: Optional[int], loglevel: str):
    """Start Celery worker."""
    # Initialize environment
    initialize_worker_environment()
    
    # Parse queues
    queue_list = [q.strip() for q in queues.split(',')]
    
    # Start worker
    manager = WorkerManager(celery_app)
    try:
        manager.start_worker(
            queues=queue_list,
            concurrency=concurrency,
            loglevel=loglevel
        )
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping worker")
        manager.stop_worker()
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
def flower():
    """Start Flower monitoring dashboard."""
    logger.info("Starting Flower monitoring dashboard")
    
    try:
        from flower.command import FlowerCommand
        flower_app = FlowerCommand()
        flower_app.run_from_argv("flower", [
            f"--broker={settings.celery.broker_url}",
            "--port=5555",
            "--basic_auth=admin:secret123",  # Change in production
        ])
    except ImportError:
        logger.error("Flower not installed. Run: pip install flower")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Flower failed to start: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option('--queue', '-q', default="health", help='Queue name for health check')
def health():
    """Run health check task."""
    logger.info("Running health check")
    
    try:
        from ..workers.tasks import health_check_task
        result = health_check_task.delay()
        response = result.get(timeout=30)
        
        if response.get("status") == "healthy":
            logger.info("Health check passed")
            click.echo("✅ Worker is healthy")
        else:
            logger.warning(f"Health check failed: {response}")
            click.echo(f"❌ Worker is unhealthy: {response}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        click.echo(f"❌ Health check error: {e}")
        sys.exit(1)


@cli.command()
def inspect():
    """Inspect worker status."""
    logger.info("Inspecting worker status")
    
    inspect = celery_app.control.inspect()
    
    # Active tasks
    active = inspect.active()
    if active:
        click.echo("Active tasks:")
        for worker, tasks in active.items():
            click.echo(f"  {worker}: {len(tasks)} tasks")
    else:
        click.echo("No active tasks")
    
    # Registered tasks
    registered = inspect.registered()
    if registered:
        click.echo("\nRegistered tasks:")
        for worker, tasks in registered.items():
            click.echo(f"  {worker}: {len(tasks)} tasks")
    
    # Worker stats
    stats = inspect.stats()
    if stats:
        click.echo("\nWorker stats:")
        for worker, stat in stats.items():
            click.echo(f"  {worker}:")
            click.echo(f"    Pool: {stat.get('pool', {}).get('max-concurrency', 'unknown')}")
            click.echo(f"    Total: {stat.get('total', 'unknown')}")


def main():
    """Main entry point for worker CLI."""
    cli()


if __name__ == "__main__":
    main()