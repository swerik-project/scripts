#!/usr/bin/env python3
"""
annotate titles
"""
from pyriksdagen.args import (
    fetch_parser,
    impute_args,
)
from pyriksdagen.io import (
    parse_tei,
    write_tei,
)
from tqdm import tqdm
import regex as re


#                                (?P<fullname>\b[A-ZÅÄÖ][a-zåäö]+(?:\s+[A-ZÅÄÖ][a-zåäö]+)+\b)

def main(args):
    salutation = r"""
                        \b(((g|G)re(f)?ve){e<=1}|
                        (((F|f)ri)?(H|h)err(e|ar)?){e<=1}|
                        (H|h)r|
                        (?i:fru){e<=1}|
                        (?i:fröken){e<=1})\b
                """
    subjkw = r"""
                        \b((O|o)m|
                        (A|a)ngående|
                        i(:)?\sanledning)\b
            """

    pat = re.compile(rf"""
                    ^(\S\s)?
                    (?P<number>
                        (
                            (N(:)?(r|o)(\.)?){{e<=1}}\s
                            .{{1,3}}(\.)?\s
                        )?
                    )
                    (?:(?P<av>A(v|f)){{e<=1}}\s)?
                    (?(av)
                        (?:
                            (?P<salutation>{salutation}).*
                            |
                            (
                                (?P<fullname>
                                    \b(?:
                                           [A-ZÅÄÖ]{{1,2}}\.?|
                                           [A-ZÅÄÖ][a-zåäö]+
                                    )
                                    (?:\s+
                                        (?:
                                             [A-ZÅÄÖ]{{1,2}}\.?|
                                             [A-ZÅÄÖ][a-zåäö]+
                                        )
                                    )+\b
                                )
                                .*
                                (?P<subjkw>{subjkw})
                                .*
                            )
                        )
                        |
                        (?P<salutation>{salutation}).*(?P<subjkw>{subjkw}).*
                    )
                    """, re.VERBOSE)


    if args.test:
        test_cases = [
                # Should match
                "4f Ola Lasson: Om fattigunderstöd af allmänna medel för den som,",
                "Af RK. P. Arnoldson: Om ändring af 28 $ 1 mom. i Regerings- formen.",
                "N:o 27. Af herr J. Anderson i Tenhult, om höjd tull å sulläder m. m.",
                "Av Fru Mróz Om saken",
                "Af Fru Markowska Angående saken",
                "Av Hr Kozławski komer en bóbr, krwa.",
                "Fru Om saken",
                "Af Margaret Thatcher om",
                # Shouldn't match
                "-----------------",
                "Om saken direkt",
                "av Idi Amin",
                "af de tafel valde het boek om",
                "herr Zdzisław Kosłąwski såg en bóbr",
            ]
        print(pat.pattern)
        for tc in test_cases:
            m = None
            m = pat.match(tc)
            if m:
                print("matched: ", tc)
            else:
                print("didn't match:", tc)

    else:
        args.motions = [_ for _ in args.motions if _.split('/')[2] not in ["fört", "reg"]]
        for motion in tqdm(args.motions):
            write = False
            root, ns = parse_tei(motion)
            Ps = root.findall(f".//{ns['tei_ns']}p")
            for p in Ps:
                if p.text is not None:
                    t = ' '.join([_.strip() for _ in p.text.splitlines() if _.strip() != ''])
                    m = pat.search(t)
                    if m:
                        p.attrib["type"] = "titleString"
                        write = True

            if write:
                write_tei(root, motion)




if __name__ == '__main__':
    parser = fetch_parser("motions", docstring=__doc__)
    parser.add_argument("--test", action='store_true')
    args = impute_args(parser.parse_args())
    main(args)
