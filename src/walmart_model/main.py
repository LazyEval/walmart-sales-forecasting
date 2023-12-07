import typer

from walmart_model.components.sales_record_loader import SalesRecordLoader


def main():
    loader = SalesRecordLoader(file_name="transactions_data.csv")
    sales_records = loader.get_sales_records()
    print(next(sales_records))


if __name__ == "__main__":
    typer.run(main)
