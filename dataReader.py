import csv


def read_dataset(file_name: str) -> list[list]:
    file = open("data/" + file_name, "r")
    csvreader = csv.reader(file)
    # remove header
    next(csvreader)

    rows = []
    for row in csvreader:
        row = process_row(row)
        rows.append(row)

    file.close()

    return rows


def process_row(row: list[str]) -> (list[str], list[int]):
    row[0] = "M" if row[0] == "male" else "F"

    row[1] = row[1][6]

    if row[2] == "some high school":
        row[2] = "shs"
    elif row[2] == "high school":
        row[2] = "hs"
    elif row[2] == "associate's degree":
        row[2] = "ad"
    elif row[2] == "some college":
        row[2] = "sc"
    elif row[2] == "bachelor's degree":
        row[2] = "bd"
    elif row[2] == "master's degree":
        row[2] = "md"

    row[3] = "f/r" if row[3] == "free/reduced" else "s"

    row[4] = "n" if row[4] == "none" else "c"
    target_values = [int(row[i]) for i in range(5, 8)]

    return row[:5], target_values


if __name__ == "__main__":
    print("Ran from dataReader.py")

    data = read_dataset("StudentsPerformance.csv")
    [print(row) for row in data]

