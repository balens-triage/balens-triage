import requests
import logging

logger = logging.getLogger(__name__)


def get_email(username, token):
    gh_api_url = 'https://api.github.com/users/{username}/events/public?access_token={secret}'.format(
        username=username, secret=token)

    fullname = 'N/A'
    email = 'N/A'

    try:
        r = requests.get(gh_api_url)
        r.raise_for_status()

        gh_public_events = r.json()
        if isinstance(gh_public_events, dict):
            if gh_public_events.get('message') and gh_public_events.get('message') == 'Not Found':
                return email, fullname

        for event in gh_public_events:
            if event.get('payload').get('commits'):
                commits = event.get('payload').get('commits')
                for commit in commits:
                    if commit.get('author'):
                        fullname = commit.get('author').get('name')
                        email = commit.get('author').get('email')
                        break
    except Exception as e:
        logger.warning(e)

    return email, fullname
