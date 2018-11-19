from src.tools.mappers.bug_vector import get_bug_vector


def get_events_per_developer(bugs):
    events_per_developer = {}
    for bug in bugs:
        developer = bug['assignee_email']
        bug_id = bug['id']

        history = bug['history']
        resolution = bug['resolution']
        idx = -1

        for assign in history:
            idx += 1

            # register this event in the dictionary
            start = assign['date']
            end = start
            final_fixer = False
            if assign['email'] == developer or idx == (len(history) - 1):
                # this is final fixer, check resolution 'FIXED' event for end date

                for resolut in resolution:
                    # if resolution is WONTFIX start date and end date are same
                    if resolut['status'].find('FIXED') > -1:
                        # change start time stamp to assignment date for this user
                        end = resolut['time']
                        final_fixer = True
                        break

            else:
                end = history[idx + 1]['date']

            event = (start, end, bug_id, final_fixer)

            if not events_per_developer.get(assign['email']):
                events_per_developer[assign['email']] = []

            events_per_developer[assign['email']].append(event)

    return events_per_developer


def get_fix_times_per_developer(bugs):
    events_per_developer = {}
    for bug in bugs:
        developer = bug['assignee_email']
        bug_vector = get_bug_vector(bug)

        history = bug['history']
        resolution = bug['resolution']
        idx = -1

        for assign in history:
            idx += 1

            if assign['email'] == developer or idx == (len(history) - 1):
                start = assign['date']

                for resolut in resolution:
                    # if resolution is WONTFIX start date and end date are same
                    if resolut['status'].find('FIXED') > -1:
                        # change start time stamp to assignment date for this user
                        end = resolut['time']
                        break
                else:
                    end = bug['modified_at']

                event = (((end - start).total_seconds() // (3600 * 24)) + 1, bug_vector, bug['id'])

                if not events_per_developer.get(assign['email']):
                    events_per_developer[assign['email']] = []

                events_per_developer[assign['email']].append(event)

    return events_per_developer
