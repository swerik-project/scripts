import argparse, os, sys, warnings


def alto_args(_file_desc):
    warnings.warn("use Pyriksdagen instead.", DepreciationWarning, stacklevel=2)
    parser = argparse.ArgumentParser(description=_file_desc)
    parser.add_argument("-s", "--start", type=int, default=None, help="start year")
    parser.add_argument("-e", "--end", type=int, default=None, help="end year")
    parser.add_argument("-y", "--year", type=int, default=None, help="Single year")
    parser.add_argument("--motionspath", type=str, default="riksdagen-motions/data")
    return parser


def verify_alto_args(args):
    warnings.warn("use Pyriksdagen instead.", DepreciationWarning, stacklevel=2)
    if args.year is not None and (args.start is not None or args.end is not None):
        print("Use -y by itself or -s and -e")
        sys.exit()
    elif (args.start is None or args.end is None) and args.start != args.end:
        print("Set -s and -e together, else use -y for one year")
        sys.exit()
    else:
        return args


def list_years(args, type="motions"):
    args = vars(args)
    types = {}
    if args.get("motionspath")is not None:
        types["motions"] = args['motionspath']
    if args.get("altopath") is not None:
        types["alto"] = args['altopath']

    if args.get("year") is not None:
        years = [args['year']]
    else:
        _range = [_ for _ in os.listdir(types[type]) if os.path.isdir(f"{types[type]}/{_}") and _ not in ["fort", "reg"]]
        years = sorted([_ for _ in _range if args['start'] <= int(_[:4]) <= args['end']])
    return years
