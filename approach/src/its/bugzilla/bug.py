import re
from datetime import datetime


def transform_bug(bug):
    result = {
        'id': bug['id'],
        'title': bug['summary'],
        'assignee': bug['assigned_to_detail']['real_name'],
        'assignee_email': bug['assigned_to'],
        'cc_list': bug['cc'],
        'status': bug['status'] + ' ' + bug['resolution'],
        'keywords': bug['keywords'],
        'created_at': bug['creation_time'],
        'modified_at': bug['last_change_time'],
        'severity': bug['severity'],
        'product': bug['product'],
        'component': bug['component']
    }

    if bug.get('comments') and len(bug['comments']) > 0:
        result['summary'] = bug['comments'][0]['text']
    else:
        result['summary'] = 'N/A'

    return result


class BugzillaBug:
    def __init__(self, **kwargs):
        self._bzapi = kwargs.get('bzapi', None)
        self._bug_id = kwargs.get('bug_id', None)
        self._bug = kwargs.get('bug', None)
        self._dict = {}

        if self._bug:
            self._exists = True
        else:
            self._exists = False
            self._bug = self.fetch()

        if self._exists:
            self._dict = self._bug.__dict__

    def fetch(self):
        try:
            response = self._bzapi.getbug(self._bug_id)
            self._exists = True
            return response
        except Exception as ex:
            self._exists = False

    def get_bug_dict(self):
        return self._dict

    def exists(self):
        return self._exists

    def has_data(self):
        return hasattr(self, '_dict')

    def date(self):
        if self.has_data():
            return datetime.strptime(self._dict['creation_time'], '%Y-%m-%dT%H:%M:%SZ')
        else:
            return None

    def get_transformed(self):
        return transform_bug(self._dict)

    def getcomments(self):
        try:
            # TODO investigate spurious token errors
            comments = self._bug.getcomments()
            self._dict['comments'] = comments
        except Exception as e:
            self._dict['comments'] = []
        return self._dict['comments']

    def _gethistory(self):
        try:
            res = self._bug.get_history_raw()
            self._dict['history'], self._dict['resolutions'] = history_to_tossings(res)
        except Exception as e:
            self._dict['history'] = []
            self._dict['resolutions'] = []

    def gethistory(self):
        if 'history' not in self._dict:
            self._gethistory()

        return self._dict['history']

    def getresolutions(self):
        if 'resolutions' not in self._dict:
            self._gethistory()

        return self._dict['resolutions']


# 7 total occurences in the dataset, so we need to handle it
# https://bugzilla.mozilla.org/rest/bug/43708/history
def clean_email(email):
    return re.sub(r'[^a-zA-Z0-9_@.-]', '', email)


def history_to_tossings(api_response):
    assignments = []
    resolutions = []

    for bug in api_response['bugs']:
        bug_id = bug['id']
        for event in bug['history']:
            when = event['when']
            for change in event['changes']:
                if change['added'] != change['removed']:
                    if change['field_name'] == 'assigned_to':
                        assignments.append({
                            'bug_id': bug_id,
                            'time': when,
                            'email': clean_email(change['added'])
                        })
                    elif change['field_name'] == 'status':
                        resolutions.append({
                            'bug_id': bug_id,
                            'time': when,
                            'status': clean_email(change['added'])
                        })

    return assignments, resolutions
