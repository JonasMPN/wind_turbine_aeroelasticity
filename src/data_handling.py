def exists_already(df, **kwargs) -> bool:
    if df.query(order_to_query(kwargs)).empty:
        return False
    else:
        return True


def slice_results(df, **kwargs):
    for parameter, value in kwargs.items():
        df = df[df[parameter]==value]
    return df


def order_to_query(order: dict, negate_order: bool=False) -> str:
    compare_by = "==" if not negate_order else "!="
    query = str()
    for param, value in order.items():
        if type(value) in [str, list]:
            query += f"{param}{compare_by}'{value}' and "
        else:
            query += f"{param}{compare_by}{value} and "
    return query[:-5]
