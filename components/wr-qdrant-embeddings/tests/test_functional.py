import unittest
from datadirtest import DataDirTester


class TestComponent(unittest.TestCase):
    """Functional tests for Qdrant Writer."""

    def test_all(self):
        """Test the default case."""
        print("\nRunning all tests...")
        tester = DataDirTester()
        tester.run()
        print("test_all finished")


if __name__ == "__main__":
    unittest.main()
