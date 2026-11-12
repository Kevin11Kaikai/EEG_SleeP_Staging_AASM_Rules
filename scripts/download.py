"""Show official acquisition instructions; never auto-download the full dataset."""
if __name__=="__main__":
    print("Use an already downloaded HMC v1.1 copy whenever available.")
    print("Official data: https://physionet.org/content/hmc-sleep-staging/1.1/")
    print("Verify files against data/SHA256SUMS.txt; keep PSG/scoring files outside Git.")
    print("This entry point prints instructions only and transfers no data.")
