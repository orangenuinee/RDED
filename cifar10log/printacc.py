import glob
import re
import numpy as np
import os

def process_log_files():
    for file_path in glob.glob("*.log"):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        accuracies = []
        test_mode = False
        
        for line in lines:
            line = line.strip()
            if line.startswith("TEST:"):
                test_mode = True
                continue
            if test_mode:
                match = re.search(r'Top-1 err = (\d+\.\d+)', line)
                if match:
                    err = float(match.group(1))
                    acc = 100 - err
                    accuracies.append(acc)
                test_mode = False
        
        if accuracies:
            mean = np.mean(accuracies)
            std = np.std(accuracies, ddof=1)
            result = f"{mean:.2f}±{std:.2f}"
        else:
            result = "No valid test data"
        
        file_name = os.path.basename(file_path)
        print(f"{file_name}: {result}")

if __name__ == "__main__":
    process_log_files()