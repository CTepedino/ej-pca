import csv


def read_dataset(filename):
    with open(filename, "r") as f:
        f.readline()
        reader = csv.reader(f)
        data = []
        for line in reader:
            data.append({
                "Country": line[0],
                "Area": int(line[1]),
                "GDP": int(line[2]),
                "Inflation": float(line[3]),
                "Life.expect": float(line[4]),
                "Military": float(line[5]),
                "Pop.growth": float(line[6]),
                "Unemployment": float(line[7])
            })
    return data

if __name__ == "__main__":
    print(read_dataset("europe.csv"))
