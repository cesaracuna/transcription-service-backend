"""
Script to check Celery worker status and queue health
"""
from src.celery_app import celery_app
import json

def check_celery_status():
    """Check Celery workers and queues status"""

    print("=" * 80)
    print("CELERY STATUS CHECK")
    print("=" * 80)

    # 1. Check active workers
    print("\n1. ACTIVE WORKERS:")
    inspect = celery_app.control.inspect()

    stats = inspect.stats()
    if stats:
        print(f"   Found {len(stats)} active worker(s):")
        for worker_name, worker_stats in stats.items():
            print(f"   - {worker_name}")
            print(f"     * Pool: {worker_stats.get('pool', {}).get('implementation', 'N/A')}")
            print(f"     * Max concurrency: {worker_stats.get('pool', {}).get('max-concurrency', 'N/A')}")
    else:
        print("   ⚠️  No active workers found!")

    # 2. Check active tasks
    print("\n2. ACTIVE TASKS:")
    active = inspect.active()
    if active:
        total_active = sum(len(tasks) for tasks in active.values())
        print(f"   Total active tasks: {total_active}")
        for worker_name, tasks in active.items():
            if tasks:
                print(f"   Worker: {worker_name}")
                for task in tasks:
                    print(f"     - {task['name']} (ID: {task['id'][:8]}...)")
                    print(f"       Args: {task.get('args', [])}")
    else:
        print("   No active tasks")

    # 3. Check scheduled tasks
    print("\n3. SCHEDULED TASKS:")
    scheduled = inspect.scheduled()
    if scheduled:
        total_scheduled = sum(len(tasks) for tasks in scheduled.values())
        print(f"   Total scheduled tasks: {total_scheduled}")
        for worker_name, tasks in scheduled.items():
            if tasks:
                print(f"   Worker: {worker_name}")
                for task in tasks:
                    print(f"     - {task['request']['name']}")
    else:
        print("   No scheduled tasks")

    # 4. Check reserved tasks
    print("\n4. RESERVED TASKS:")
    reserved = inspect.reserved()
    if reserved:
        total_reserved = sum(len(tasks) for tasks in reserved.values())
        print(f"   Total reserved tasks: {total_reserved}")
        for worker_name, tasks in reserved.items():
            if tasks:
                print(f"   Worker: {worker_name}")
                for task in tasks:
                    print(f"     - {task['name']}")
    else:
        print("   No reserved tasks")

    # 5. Check registered tasks
    print("\n5. REGISTERED TASKS:")
    registered = inspect.registered()
    if registered:
        for worker_name, tasks in registered.items():
            print(f"   Worker: {worker_name}")
            task_list = [t for t in tasks if t.startswith('tasks.')]
            for task in sorted(task_list):
                print(f"     - {task}")
    else:
        print("   ⚠️  No registered tasks found!")

    # 6. Check active queues
    print("\n6. ACTIVE QUEUES:")
    active_queues = inspect.active_queues()
    if active_queues:
        for worker_name, queues in active_queues.items():
            print(f"   Worker: {worker_name}")
            for queue in queues:
                print(f"     - Queue: {queue['name']}")
                print(f"       Exchange: {queue['exchange']['name']}")
                print(f"       Routing key: {queue['routing_key']}")
    else:
        print("   ⚠️  No active queues found!")

    # 7. Summary
    print("\n7. SUMMARY:")
    print("=" * 80)

    if stats:
        print(f"[OK] {len(stats)} worker(s) online")
    else:
        print("[FAIL] No workers online - Start Celery worker!")

    if active:
        total_active = sum(len(tasks) for tasks in active.values())
        print(f"[OK] {total_active} task(s) currently processing")

    if registered and any(tasks for tasks in registered.values()):
        print("[OK] Tasks are registered and ready")
    else:
        print("[FAIL] No tasks registered - Check task imports!")

    if active_queues:
        print("[OK] Queues are active and processing")
        # Check if using correct queue
        for worker_name, queues in active_queues.items():
            queue_names = [q['name'] for q in queues]
            if 'default' not in queue_names and 'celery' in queue_names:
                print("[WARNING] Worker is listening to 'celery' queue but tasks are sent to 'default' queue!")
                print("          Restart worker with: celery -A src.celery_app worker -Q default --loglevel=info")
    else:
        print("[FAIL] No queues active - Check broker connection!")

    print("=" * 80)

if __name__ == "__main__":
    try:
        check_celery_status()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nMake sure:")
        print("  1. Redis/RabbitMQ broker is running")
        print("  2. Celery worker is running")
        print("  3. Network connectivity is available")