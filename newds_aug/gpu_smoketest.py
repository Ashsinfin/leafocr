import time, sys, importlib.util, cv2
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices("GPU"))
spec = importlib.util.spec_from_file_location("app", "app.py")
app = importlib.util.module_from_spec(spec); spec.loader.exec_module(app)
img = cv2.cvtColor(cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "846_1_patch000.png"), cv2.COLOR_BGR2RGB)
for run in (1, 2):
    t = time.time(); last = None
    for last in app.run_pipeline(img): pass
    print(f"run {run}: {time.time()-t:.1f}s")
print("TEXT:\n", last[app.OUT_KEYS.index("plain")][:800])
