"""Run the whole project in order. From the project folder:  python run.py"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "projectpipeline")
STEPS = ["data_prep.py", "compare_models.py", "train_evaluate.py", "room_recommendations.py", "make_figures.py"]

for step in STEPS:
    print(f"\n===== {step} =====", flush=True)
    subprocess.run([sys.executable, os.path.join(CODE, step)], check=True, cwd=CODE)
print("\nDone. See results/ and README.md")
