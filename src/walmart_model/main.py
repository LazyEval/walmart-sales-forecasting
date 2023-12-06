import typer

from walmart_model.components.csv_loader import CSVLoader


def main():
    loader = CSVLoader(file_name="transactions_data_sampled")
    transactions = loader.get_transactions()
    print(transactions[:5])


if __name__ == "__main__":
    typer.run(main)
