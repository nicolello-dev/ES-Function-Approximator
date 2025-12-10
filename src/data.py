from typing import List, Tuple


def load_data(data_file_name: str) -> List[Tuple[float, float]]:
    """
    Loads the data from the input file

    :return: Returns a two-dimensional list containing [input, output] pairs
    :rtype: List[Tuple[float, float]]
    """

    try:
        with open(data_file_name, "r") as file:
            raw_data = file.readlines()
    except FileNotFoundError:
        print(f"Data file '{data_file_name}' not found.")
        exit(1)

    # Array of [input, output] pairs
    data = []

    for line in raw_data:
        values = line.strip().split(" ")
        assert len(values) >= 2
        temp = tuple((float(values[0]), float(values[-1])))
        data.append(temp)

    return data
