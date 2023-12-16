import typer

from walmart_model.components.item_splitter import ItemSplitter
from walmart_model.components.transaction_processor import TransactionProcessor

TRANSACTIONS_FILE_NAME = "transactions_data.csv"
FORECAST_HORIZON = 28


def main():
    processor = TransactionProcessor(
        file_name=TRANSACTIONS_FILE_NAME,
        splitter=ItemSplitter(forecast_horizon=FORECAST_HORIZON),
    )
    processor.save_items()


if __name__ == "__main__":
    typer.run(main)
