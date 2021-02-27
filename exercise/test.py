from tqdm import tqdm

bar = tqdm(["a", "b", "c", "d"])
for char in bar:
    bar.set_description("Processing %s" % char)