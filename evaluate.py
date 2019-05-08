"""
Reported results from See et al. (2017)

Pointer Generator:
    ROUGE-1 (F1): 36.44
    ROUGE-2 (F1): 15.66
    ROUGE-L (F1): 33.42

Pointer Generator + Coverage:
    ROUGE-1 (F1): 39.53
    ROUGE-2 (F1): 17.28
    ROUGE-L (F1): 36.38

Example:
    python rouge_evaluate.py test_output/pointer-gen test_output/reference
    python rouge_evaluate.py test_output/pointer-gen-cov test_output/reference
"""
import sys
from glob import glob
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import tempfile
import rouge
import pyrouge

#pylint: disable=invalid-name, too-few-public-methods

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class Rouge:
    """Make `Rouge155` input/output compatible with py-rouge's `Rouge`"""
    def __init__(self, perl=False):
        self.perl = perl

    @staticmethod
    def _get_scores_perl(hypothesis, references):
        # get path to rouge based on __file__ path
        file_path = os.path.dirname(os.path.realpath(__file__))
        r = pyrouge.Rouge155(os.path.join(file_path, "ROUGE-1.5.5"))
        ref_dir = tempfile.mkdtemp()
        hyp_dir = tempfile.mkdtemp()
        for idx, (ref, hyp) in enumerate(zip(references, hypothesis)):
            ref_file = os.path.join(ref_dir, "%06d_reference.txt" % idx)
            hyp_file = os.path.join(hyp_dir, "%06d_hypothesis.txt" % idx)
            with open(ref_file, "w") as rf, open(hyp_file, "w") as hf:
                rf.write(ref)
                hf.write(hyp)

        # model is gold standard, system is hypothesis
        r.model_dir = ref_dir
        r.system_dir = hyp_dir
        r.model_filename_pattern = "#ID#_reference.txt"
        # pylint: disable=anomalous-backslash-in-string
        r.system_filename_pattern = "(\d+)_hypothesis.txt"

        output = r.convert_and_evaluate()
        output = r.output_to_dict(output)
        return {
            "rouge-1": {
                "p": output["rouge_1_precision"],
                "r": output["rouge_1_recall"],
                "f": output["rouge_1_f_score"],
            },
            "rouge-2": {
                "p": output["rouge_2_precision"],
                "r": output["rouge_2_recall"],
                "f": output["rouge_2_f_score"],
            },
            "rouge-l": {
                "p": output["rouge_l_precision"],
                "r": output["rouge_l_recall"],
                "f": output["rouge_l_f_score"],
            },
        }

    @staticmethod
    def _get_scores_python(hypothesis, references):
        """Note: py-rouge mixes up recall/precision"""
        return rouge.Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=False,
            apply_avg=True,
            alpha=0.5,  # Default F1_score
            stemming=True,
            ensure_compatibility=True,
        ).get_scores(hypothesis, references)

    def get_scores(self, hypothesis, references):
        """
        Get rouge scores as a dict of format:
        {"rouge-1": {"p": 0.5,
                     "r": 0.3,
                     "f": 0.4},
         "rouge-2: ...,
         "rouge-l: ...
        }

        :param references: A list of reference summaries
        :param hypothesis: A list of corresponding summaries
        :returns: A dictionary with the rouge scores
        """
        if not self.perl:
            return self._get_scores_python(hypothesis, references)

        with suppress_stdout_stderr():
            return self._get_scores_perl(hypothesis, references)


def print_scores(scores):
    """
    Pretty print ROUGE F-1 scores to stdout

    :param scores: the scores output by `Rouge().get_scores`
    """
    types = ["rouge-1", "rouge-2", "rouge-l"]
    for t in types:
        f1 = scores[t]["f"] * 100
        print(f"\t {t.upper()} (F1): {f1:.2f}")


def rouge_evaluate(ref_folder, hyp_folder, perl=True):
    """Return rouge scores for the given `ref_folder` and `hyp_folder`"""
    ref_paths = glob(os.path.join(ref_folder, "*reference.txt"))
    references = []
    hypothesis = []
    for ref_path in ref_paths:
        hyp_path = ref_path.replace(ref_folder, hyp_folder)
        hyp_path = hyp_path.replace("reference.txt", "decoded.txt")
        with open(ref_path) as ref_file, open(hyp_path) as hyp_file:
            ref = ref_file.read()
            hyp = hyp_file.read()
            if ref and hyp:
                references.append(ref)
                hypothesis.append(hyp)

    r = Rouge(perl=perl)
    return r.get_scores(hypothesis, references)


def main():
    """Run evaluation with py-rouge and pyrouge <hyp_folder> <ref_folder>"""
    hyp_folder = sys.argv[1]
    ref_folder = sys.argv[2]
    python_scores = rouge_evaluate(ref_folder, hyp_folder, perl=False)
    print("Python (py-rouge) scores:")
    print_scores(python_scores)

    perl_scores = rouge_evaluate(ref_folder, hyp_folder, perl=True)
    print("\nPerl (pyrouge) scores:")
    print_scores(perl_scores)

if __name__ == '__main__':
    assert len(sys.argv) == 3, "args should be <hyp_folder> <ref_folder>"
    main()
