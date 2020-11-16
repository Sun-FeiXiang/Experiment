CCs = dict()
c = 0
for i in range(0, 5):
    c = c + 1
    CCs[c] = i + 8
print(CCs)

max_v, max_score = max(CCs.items())
