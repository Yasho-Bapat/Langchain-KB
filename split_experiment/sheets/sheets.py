from __future__ import print_function

import re

from split_experiment.sheets.auth import spreadsheet_service


class Spreadsheet:
    def __init__(self, spreadsheet_id: str):
        self.spreadsheet_id = spreadsheet_id

    def read_ranges(self, sheet_name: str, ranges: list[str]):
        range_names = [sheet_name + "!" + range for range in ranges]
        result = spreadsheet_service.spreadsheets().values().batchGet(
            spreadsheetId=self.spreadsheet_id, ranges=range_names).execute()
        ranges = result.get('valueRanges', [])
        print('{0} ranges retrieved.'.format(len(ranges)))
        print(ranges)
        return ranges

    def write_ranges(self, sheet_name: str, range: str, values: list[list]):
        # input validation
        def cell_to_row_col(cell: str):
            match = re.match(r"([A-Z]+)(\d+)", cell)
            if not match:
                raise ValueError(f"Invalid cell reference: {cell}")
            col_str, row_str = match.groups()
            row = int(row_str)
            col = sum((ord(char) - ord('A') + 1) * (26 ** idx) for idx, char in enumerate(reversed(col_str)))
            return row, col

        start, end = range.split(":")
        start_row, start_col = cell_to_row_col(start)
        end_row, end_col = cell_to_row_col(end)

        num_rows = end_row - start_row + 1
        num_cols = end_col - start_col + 1

        if len(values) != num_rows:
            raise ValueError(f"Number of rows in values ({len(values)}) does not match the range ({num_rows})")
        for row in values:
            if len(row) != num_cols:
                raise ValueError(f"Number of columns in values row ({len(row)}) does not match the range ({num_cols})")

        data = [{"range": f'{sheet_name}!{range}', "values": values}]

        body = {
            'valueInputOption': 'USER_ENTERED',
            'data': data
        }
        result = spreadsheet_service.spreadsheets().values().batchUpdate(
            spreadsheetId=self.spreadsheet_id, body=body).execute()
        print('{0} cells updated.'.format(result.get('totalUpdatedCells')))
