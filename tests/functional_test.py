from utils.data_utils import load_data_from_excels


class TestRules:
    def test_load_data(self):
        files = ["predata.xls", "Data.xlsx"]
        data = load_data_from_excels(files)
        print(len(data))
        pass
