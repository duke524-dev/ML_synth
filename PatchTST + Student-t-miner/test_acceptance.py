"""
Acceptance tests for ForecastEngine
"""
import unittest
import logging
from datetime import datetime, timezone, timedelta
import numpy as np

from forecast_engine import ForecastEngine
from config import HF_ASSETS, LF_CRYPTO_ASSETS, LF_EQUITY_ASSETS

logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
logger = logging.getLogger(__name__)


class TestForecastEngine(unittest.TestCase):
    """Acceptance tests for ForecastEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = ForecastEngine(artifacts_dir="artifacts")
    
    def test_hf_request_format(self):
        """Test HF request format"""
        start_time = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
        
        result = self.engine.generate_paths(
            asset="BTC",
            start_time=start_time,
            time_increment=60,
            time_length=3600,
            num_simulations=100,
        )
        
        # Check format
        self.assertIsInstance(result, tuple)
        self.assertGreaterEqual(len(result), 2)
        
        # Check first two elements
        start_timestamp, time_increment = result[0], result[1]
        self.assertIsInstance(start_timestamp, int)
        self.assertEqual(time_increment, 60)
        
        # Check paths
        paths = result[2:]
        self.assertEqual(len(paths), 100)
        
        # Check each path
        expected_points = (3600 // 60) + 1  # 61 points
        for path in paths:
            self.assertIsInstance(path, list)
            self.assertEqual(len(path), expected_points)
            
            # Check all prices are positive and finite
            for price in path:
                self.assertIsInstance(price, (int, float))
                self.assertGreater(price, 0)
                self.assertTrue(np.isfinite(price))
    
    def test_lf_crypto_request_format(self):
        """Test LF crypto request format"""
        start_time = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
        
        result = self.engine.generate_paths(
            asset="BTC",
            start_time=start_time,
            time_increment=300,
            time_length=86400,
            num_simulations=1000,
        )
        
        # Check format
        self.assertIsInstance(result, tuple)
        self.assertGreaterEqual(len(result), 2)
        
        start_timestamp, time_increment = result[0], result[1]
        self.assertIsInstance(start_timestamp, int)
        self.assertEqual(time_increment, 300)
        
        # Check paths
        paths = result[2:]
        self.assertEqual(len(paths), 1000)
        
        # Check each path
        expected_points = (86400 // 300) + 1  # 289 points
        for path in paths:
            self.assertEqual(len(path), expected_points)
            
            for price in path:
                self.assertGreater(price, 0)
                self.assertTrue(np.isfinite(price))
    
    def test_lf_equity_request_format(self):
        """Test LF equity request format"""
        start_time = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
        
        result = self.engine.generate_paths(
            asset="SPYX",
            start_time=start_time,
            time_increment=300,
            time_length=86400,
            num_simulations=1000,
        )
        
        # Check format
        self.assertIsInstance(result, tuple)
        paths = result[2:]
        self.assertEqual(len(paths), 1000)
        
        expected_points = (86400 // 300) + 1
        for path in paths:
            self.assertEqual(len(path), expected_points)
    
    def test_timestamp_correctness(self):
        """Test that timestamps are correct"""
        start_dt = datetime.now(timezone.utc) + timedelta(minutes=1)
        start_time = start_dt.isoformat()
        
        result = self.engine.generate_paths(
            asset="BTC",
            start_time=start_time,
            time_increment=300,
            time_length=86400,
            num_simulations=10,
        )
        
        start_timestamp = result[0]
        expected_timestamp = int(start_dt.timestamp())
        
        # Allow small difference due to rounding
        self.assertAlmostEqual(start_timestamp, expected_timestamp, delta=1)
    
    def test_anchor_price_applied(self):
        """Test that anchor price is applied correctly"""
        start_time = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
        
        result = self.engine.generate_paths(
            asset="BTC",
            start_time=start_time,
            time_increment=300,
            time_length=86400,
            num_simulations=10,
        )
        
        paths = result[2:]
        
        # All paths should start with the same anchor price (within rounding)
        anchor_prices = [path[0] for path in paths]
        
        # All should be close (within 1% due to rounding/formatting)
        first_anchor = anchor_prices[0]
        for anchor in anchor_prices:
            self.assertAlmostEqual(anchor, first_anchor, delta=first_anchor * 0.01)
    
    def test_runtime_performance(self):
        """Test that generation completes in reasonable time"""
        import time
        
        start_time = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
        
        start_clock = time.time()
        result = self.engine.generate_paths(
            asset="BTC",
            start_time=start_time,
            time_increment=300,
            time_length=86400,
            num_simulations=1000,
        )
        elapsed = time.time() - start_clock
        
        # Should complete in under 30 seconds on CPU
        self.assertLess(elapsed, 30.0, f"Generation took {elapsed:.2f}s, expected < 30s")
        
        # Check result is valid
        self.assertIsNotNone(result)
        self.assertEqual(len(result[2:]), 1000)
    
    def test_all_assets(self):
        """Test that all supported assets work"""
        start_time = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
        
        all_assets = list(HF_ASSETS | LF_CRYPTO_ASSETS | LF_EQUITY_ASSETS)
        
        for asset in all_assets:
            try:
                result = self.engine.generate_paths(
                    asset=asset,
                    start_time=start_time,
                    time_increment=300,
                    time_length=86400,
                    num_simulations=10,
                )
                
                self.assertIsNotNone(result)
                self.assertGreaterEqual(len(result), 2)
                
            except Exception as e:
                self.fail(f"Failed for asset {asset}: {e}")
    
    def test_edge_cases(self):
        """Test edge cases"""
        start_time = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
        
        # Test with minimal simulations
        result = self.engine.generate_paths(
            asset="BTC",
            start_time=start_time,
            time_increment=300,
            time_length=86400,
            num_simulations=1,
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result[2:]), 1)
        
        # Test with different time increments
        for increment in [60, 300]:
            result = self.engine.generate_paths(
                asset="BTC",
                start_time=start_time,
                time_increment=increment,
                time_length=increment * 10,  # 10 steps
                num_simulations=5,
            )
            
            self.assertIsNotNone(result)
            expected_points = 10 + 1
            self.assertEqual(len(result[2][0]), expected_points)


def run_acceptance_tests():
    """Run all acceptance tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestForecastEngine)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_acceptance_tests()
    exit(0 if success else 1)
