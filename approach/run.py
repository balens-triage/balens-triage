import argparse
import os
from src import *

# Set the execution environment to the dir of this file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

parser = argparse.ArgumentParser(description='Run the Python Bug Triaging Recommender.')

parser.add_argument('-f', '--fetch', action='store_true')
parser.add_argument('-u', '--update', action='store_true', help='Update stored bugs')
parser.add_argument('-t', '--train', action='store_true')
parser.add_argument('-dl', '--datalimit',
                    help="""Limit training data points to this number. Useful for debugging.""")
parser.add_argument('-mt', '--maxthreads',
                    help="""Maximum number of threads to use for non training 
                    tasks such as fetching and updating""")
parser.add_argument('-s', '--stackoverflow', action='store_true',
                    help='load the stackoverflow data into the database')
parser.add_argument('-c', '--config', default='./config.json', help='path to the config file')
parser.add_argument('-cl', '--cachelevel', default='MEDIUM', help='level of caching.')
parser.add_argument('-cb', '--cachebackend', default='LOCAL',
                    help='Caching backend to use. Available: Local, S3.')

parser.add_argument('-p', '--pipeline',
                    help='Overwrite the pipeline field on all projects with csv')

args = parser.parse_args()

main = Main(args.config,
            pipeline=args.pipeline,
            cache_level=args.cachelevel,
            cache_backend=args.cachebackend,
            datalimit=args.datalimit,
            maxthreads=args.maxthreads)

if args.fetch:
    main.fetch()

if args.train:
    main.train()

if args.update:
    main.update()

if args.stackoverflow:
    main.load_stackoverflow()

if not args.train and not args.fetch and not args.stackoverflow:
    parser.print_help()
