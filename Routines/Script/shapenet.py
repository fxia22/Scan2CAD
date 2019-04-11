with open('unique_cads.csv') as f:
    for line in f:
        print(' '.join(line.strip().split(',')))
