from __static__ import CheckedList, int64


ROWS = CheckedList[CheckedList[int64]]([CheckedList[int64]([1, 2, 3])])


def first_row_value(rows=ROWS) -> int64:
    row: CheckedList[int64]
    for row in rows:
        return row[0]
    return 0
