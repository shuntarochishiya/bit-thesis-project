
import time
import unittest

from engine.dag_scheduler import DAGScheduler


class TestParallelScheduler(unittest.TestCase):

    def setUp(self):
        self.scheduler = DAGScheduler()

    @staticmethod
    def slow_task(task):
        time.sleep(0.3)
        return task["task_id"]

    def test_parallel_execution(self):
        tasks = [
            {"task_id": "task_a", "parallel_safe": True},
            {"task_id": "task_b", "parallel_safe": True},
            {"task_id": "task_c", "parallel_safe": True},
        ]

        start = time.perf_counter()

        results = self.scheduler.execute_level(
            tasks,
            self.slow_task,
            max_workers=3
        )

        elapsed = time.perf_counter() - start

        self.assertEqual(len(results), 3)
        self.assertLess(elapsed, 0.75)

        print(f"\nParallel execution: {elapsed:.3f}s")

    def test_sequential_execution(self):
        tasks = [
            {"task_id": "task_a", "parallel_safe": True},
            {"task_id": "task_b", "parallel_safe": False},
            {"task_id": "task_c", "parallel_safe": True},
        ]

        start = time.perf_counter()

        results = self.scheduler.execute_level(
            tasks,
            self.slow_task,
            max_workers=3
        )

        elapsed = time.perf_counter() - start

        self.assertEqual(len(results), 3)
        self.assertGreaterEqual(elapsed, 0.9)

        print(f"\nSequential execution: {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
