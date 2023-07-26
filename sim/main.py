import numpy as np
import SNG
import RNS
import PCC

def main():
    """THIS SHOULD BE THE MAIN ENTRY POINT FOR EVERYTHING!"""
    print(SNG.sng(np.array([0.75, 0.25]), 16, 4, RNS.lfsr, PCC.CMP))
    
if __name__ == "__main__":
    main()