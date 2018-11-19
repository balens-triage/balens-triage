import hashlib

import psycopg2.extras

from django.http import HttpResponse
from django.template import loader

from webtriage.main import instance

conn = instance.get_connection()


def get_its_and_url(namespace):
  first, second, third = namespace.split('_')

  if first == 'github':
    return first, 'https://github.com/' + first + '/' + second
  else:
    return 'bugzilla', 'https://' + first + '.' + second + '.' + third


def get_datasets():
  results = {}

  with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
    cursor.execute("""select count(*), ns from issues group by ns""")
    issues = cursor.fetchall()

    for issue in issues:
      results[issue['ns']] = {}

      its, url = get_its_and_url(issue['ns'])

      results[issue['ns']]['its'] = its
      results[issue['ns']]['url'] = url
      results[issue['ns']]['issues'] = issue['count']

    cursor.execute("""select count(*), ns from commit_logs group by ns""")
    issues = cursor.fetchall()

    for issue in issues:
      results[issue['ns']]['logs'] = issue['count']

    return results


def bugzilla_avatar(email):
  return hashlib.md5(email.encode()).hexdigest()


def transform(predictions, namespace):
  def map_pred(pred):
    userid, cost, accuracy, fix_time = pred

    avatar_url = bugzilla_avatar(userid)
    profile_link = 'https://bugzilla.mozilla.org/user_profile?login=' + userid

    if 'github' in namespace:
      avatar_url = 'https://avatars.githubusercontent.com/' + userid
      profile_link = 'https://github.com/' + userid

    return {
      'avatar_url': avatar_url,
      'userid': userid,
      'accuracy': accuracy,
      'fix_time': fix_time,
      'profile_link': profile_link
    }

  return [map_pred(pred) for pred in predictions]


def index(request, namespace):
  template = loader.get_template('predict.html')

  context = {}

  if request.method == 'POST':
    title = request.POST.get('title')
    summary = request.POST.get('title')

    try:
      predictions = instance.balens_predict(namespace, title, summary)
      print(predictions)
      context['result'] = transform(predictions, namespace)
    except Exception as e:
      context['error'] = e

  return HttpResponse(template.render(context, request))
