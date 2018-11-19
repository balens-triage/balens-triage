def load_data(storage):
    data_rows = storage.load_bugs_and_comments()

    developer_ids = [row['assignee_email'] for row in data_rows]

    return {
        'data': data_rows,
        'target': [row['assignee_email'] for row in data_rows],
        'storage': storage,
        'target_names': list(set(developer_ids)),
        'vectorized': False
    }
