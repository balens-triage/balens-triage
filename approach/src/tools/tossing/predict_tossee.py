DEVELOPER_INACTIVE_AFTER = 100  # days


def bhattacharya_predict(graph, bug, developer_id, developer_activity):
    edges = graph.out_edges(developer_id, data=True)

    b_modified_at = bug['modified_at']
    b_component = bug['component']

    def build_dev(edge):
        last_active_date = developer_activity[edge[1]]
        delta_days = (b_modified_at - last_active_date).days

        developer = {
            'email': edge[1],
            'proba': edge[2]['weight'],
            'comp_match': b_component in edge[2]['components'],
            'active': delta_days <= DEVELOPER_INACTIVE_AFTER
        }

        return developer, rank_dev(developer)

    developers = [build_dev(edge) for edge in edges]

    return sorted(developers, key=lambda dev: dev[1], reverse=True)[0]


def predict_tossee(graph, bug, developer_id, developer_activity):
    # TODO implement own prediction
    return bhattacharya_predict(graph, bug, developer_id, developer_activity)


def rank_dev(developer):
    return developer['proba'] + (1 if developer['comp_match'] else 0) + (
        1 if developer['active'] else 0)
