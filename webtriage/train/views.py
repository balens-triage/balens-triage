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
      results[issue['ns']]['status'] = instance.model_status(issue['ns'])
      results[issue['ns']]['issues'] = issue['count']

    cursor.execute("""select count(*), ns from commit_logs group by ns""")
    issues = cursor.fetchall()

    for issue in issues:
      results[issue['ns']]['logs'] = issue['count']

    return results


def index(request):
  template = loader.get_template('datasets.html')
  context = {
    'datasets': get_datasets(),
  }

  return HttpResponse(template.render(context, request))


def train_model(request, namespace):
  if request.method == 'POST':
    instance.train_balens(namespace)
    return HttpResponse('success')
  else:
    return HttpResponse('use post on this route')
