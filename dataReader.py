import csv
from sklearn import preprocessing

labels = ["male", "female", "group A", "group B", "group C", "group D", "group E", "some high school", "high school",
          "associate's degree", "some college", "bachelor's degree", "master's degree", "free/reduced", "standard",
          "none", "completed"]
label_encoder = preprocessing.LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)


def read_dataset(file_name: str) -> list[(list[int], list[int])]:
    file = open("data/" + file_name, "r")
    csvreader = csv.reader(file)
    # remove header
    next(csvreader)

    rows = []
    for row in csvreader:
        feature, output = process_row(row)
        rows.append((feature, output))

    file.close()

    return rows


def process_row(row: list[str]) -> (list[int], list[int]):
    input_values = [labels.index(value) for value in row[:5]]
    output_values = [int(row[i]) for i in range(5, 8)]

    return input_values, output_values


if __name__ == "__main__":
    print("Ran from dataReader.py")

    _data = read_dataset("StudentsPerformance.csv")
    [print(row) for row in _data]

