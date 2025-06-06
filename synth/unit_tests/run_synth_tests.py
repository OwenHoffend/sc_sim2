import unittest
def run_all_tests():
    # Import all test modules
    import synth.unit_tests.sat_tests as SatTest
    import synth.unit_tests.COOPT_tests as COOPTTest

    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases from each module
    suite.addTests(unittest.TestLoader().loadTestsFromModule(SatTest))
    suite.addTests(unittest.TestLoader().loadTestsFromModule(COOPTTest))

    # Run all tests
    result = unittest.TextTestRunner().run(suite)
    return result