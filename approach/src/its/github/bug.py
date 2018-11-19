def map_state(state):
    if state == 'open':
        return 'UNCONFIRMED'
    elif state == 'closed':
        return 'RESOLVED FIXED'


def transform_bug(bug):
    return {
        'id': bug.id,
        'summary': bug.body,
        'title': bug.title,
        'assignee': bug.user.login,
        'cc_list': [],
        'keywords': [],
        'status': map_state(bug.state),
        'created_at': bug.created_at,
        'modified_at': bug.updated_at,
        'product': '',
        'component': ''  # TODO maybe use top level directories as features?
    }


def transform_comment(comment, bugid):
    return {
        'id': comment.id,
        'bug_id': bugid,
        'creator': comment.user.login,
        'created_at': comment.created_at,
        'text': comment.body,
        'count': 0
    }


class GithubBug:
    def __init__(self, bug):
        self._bug = bug
        self._assignments, self._resolutions = None, None

        if self._bug:
            self._exists = True
        else:
            self._exists = False

    def get_bug_dict(self):
        return self._bug

    def exists(self):
        return self._exists

    def get_transformed(self):
        return transform_bug(self._bug)

    def getcomments(self):
        try:
            comments = self._bug.get_comments()
            return [transform_comment(comment, self._bug.id)
                    for comment in comments]
        except Exception as e:
            return []

    def _gethistory(self):
        assignments = []
        resolutions = []

        events = self._bug.get_events()
        for event in events:
            if event.event == 'assigned' and event._rawData['assignee']:
                assignments.append({
                    'bug_id': self._bug.id,
                    'time': event.created_at,
                    'email': event._rawData['assignee']['login']
                })
            elif event.event == 'closed':
                resolutions.append({
                    'bug_id': self._bug.id,
                    'time': event.created_at,
                    'status': 'RESOLVED FIXED'
                })

        self._assignments = assignments
        self._resolutions = resolutions

    def gethistory(self):
        if not self._assignments:
            self._gethistory()
        return self._assignments

    def getresolutions(self):
        if not self._resolutions:
            self._gethistory()
        return self._resolutions

    def updated_at(self):
        if self._exists:
            return self._bug._updated_at.value
