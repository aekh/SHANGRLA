from dataclasses import dataclass
import math
from abc import ABC, abstractmethod

import numpy as np
import json
import csv
import types
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Collection, Iterable
from typing import Tuple, Callable
from cryptorandom.cryptorandom import int_from_hash
from cryptorandom.sample import random_permutation
from cryptorandom.sample import sample_by_index

from .NonnegMean import NonnegMean
from .Audit import Audit


class CVR:
    """
    Generic class for cast-vote records.

    The CVR class does not necessarily impose voting rules. For instance, the social choice
    function might consider a CVR that contains two votes in a contest to be an overvote.

    Rather, by default a CVR is supposed to reflect what the ballot shows, even if the ballot does not
    contain a valid vote in one or more contests.

    Class method get_vote_for returns the vote for a given candidate if the candidate is a
    key in the CVR, or False if the candidate is not in the CVR.

    This allows very flexible representation of votes, including ranked voting.

    For instance, in a plurality contest with four candidates, a vote for Alice (and only Alice)
    in a mayoral contest could be represented by any of the following:
            {"id": "A-001-01", "pool": False, "pool_group": "ABC", "phantom:: False"votes": {"mayor": {"Alice": True}}}
            {"id": "A-001-01", "votes": {"mayor": {"Alice": "marked"}}}
            {"id": "A-001-01", "votes": {"mayor": {"Alice": 5}}}
            {"id": "A-001-01", "votes": {"mayor": {"Alice": 1, "Bob": 0, "Candy": 0, "Dan": ""}}}
            {"id": "A-001-01", "votes": {"mayor": {"Alice": True, "Bob": False}}}
    A CVR that contains a vote for Alice for "mayor" and a vote for Bob for "DA" could be represented as
            {"id": "A-001-01", "votes": {"mayor": {"Alice": True}, "DA": {"Bob": True}}}

    NOTE: some methods distinguish between a CVR that contains a particular contest, but no valid
    vote in that contest, and a CVR that does not contain that contest at all. Thus, the following
    are not equivalent:
            {"id": "A-001-01", "votes": {"mayor": {}} }
            and
            {"id": "A-001-01", "votes": {} }

    Ranked votes also have simple representation, e.g., if the CVR is
            {"id": "A-001-01", "votes": {"mayor": {"Alice": 1, "Bob": 2, "Candy": 3, "Dan": ''}}}
    Then int(get_vote_for("Candy","mayor"))=3, Candy's rank in the "mayor" contest.

    CVRs can be flagged as `phantoms` to account for ballot cards not listed in the manifest using the boolean
    `phantom` attribute.

    CVRs can be assigned to a `tally_pool`, useful for the ONEAudit method or batch-level comparison audits
    using the `pool` attribute (batch-level comparison audits are not currently implemented)

    CVRs can be flagged for use in ONEAudit "pool" assorter means. When a CVR is flagged this way, the
    value of the assorter applied to the MVR is compared to the mean value of the assorter applied to the
    CVRs in the tally pool the CVR belongs to.

    CVRs can include sampling probabilities `p` and sample numbers `sample_num` (pseudo-random numbers
    to facilitate consistent sampling)

    CVRs can include a sequence number to facilitate ordering, sorting, and permuting

    Methods:
    --------

    get_vote_for:
         get_vote_for(candidate, contest_id) returns the value in the votes dict for the key `candidate`, or
         False if the candidate did not get a vote or the contest_id is not in the CVR
    has_contest: returns bool
         does the CVR have the contest?
    cvrs_to_json:
         represent CVR list as json
    from_dict: create a CVR from a dict
    from_dict_of_dicts:
         create dict of CVRs from a list of dicts
    from_raire:
         create CVRs from the RAIRE representation
    """

    def __init__(
            self,
            id: object = None,
            card_in_batch: int = None,
            votes: dict = {},
            phantom: bool = False,
            tally_pool: object = None,
            pool: bool = False,
            sample_num: float = None,
            p: float = None,
            sampled: bool = False,
    ):
        self.id = id  # identifier
        self.card_in_batch = card_in_batch  # position of the corresponding card in a physical batch. Used for ONEAudit.
        self.votes = votes  # contest/vote dict
        self.phantom = phantom  # is this a phantom CVR?
        self.tally_pool = tally_pool  # what tallying pool of cards does this CVR belong to (used by ONEAudit)?
        self.pool = pool  # pool votes on this CVR within its tally_pool?
        self.sample_num = sample_num  # pseudorandom number used for consistent sampling
        self.p = p  # sampling probability
        self.sampled = sampled  # is this CVR in the sample?

    def __str__(self) -> str:
        return (
                f"id: {str(self.id)} card_in_batch: {str(self.card_in_batch)} "
                + f"votes: {str(self.votes)}\nphantom: {str(self.phantom)} "
                + f"tally_pool: {str(self.tally_pool)} pool: {str(self.pool)} sample_num: {self.sample_num} "
                + f"p: {self.p} sampled: {self.sampled}"
        )

    def get_vote_for(self, contest_id: str, candidate: str):
        return (
            False
            if (contest_id not in self.votes or candidate not in self.votes[contest_id])
            else self.votes[contest_id][candidate]
        )

    def has_contest(self, contest_id: str) -> bool:
        return contest_id in self.votes

    def update_votes(self, votes: dict) -> bool:
        """
        Update the votes for any contests the CVR already contains; add any contests and votes not already contained

        Parameters
        ----------
        votes: dict of dict of dicts
           key is a contest id; value is a dict of votes--keys and values

        Returns
        -------
        added: bool
            True if the contest was already present; else false

        Side effects
        ------------
        updates the CVR to add the contest if it was not already present and to update the votes
        """
        added = False
        for c, v in votes.items():
            if self.has_contest(c):
                self.votes[c].update(v)
            else:
                self.votes[c] = v
                added = True
        return added

    def has_one_vote(self, contest_id: str, candidates: list) -> bool:
        """
        Is there exactly one vote among the candidates in the contest `contest_id`?

        Parameters:
        -----------
        contest_id: string
            identifier of contest
        candidates: list
            list of identifiers of candidates

        Returns:
        ----------
        True if there is exactly one vote among those candidates in that contest, where a
        vote means that the value for that key casts as boolean True.
        """
        v = np.sum(
            [
                (
                    0
                    if c not in self.votes[contest_id]
                    else bool(self.votes[contest_id][c])
                )
                for c in candidates
            ]
        )
        return True if v == 1 else False

    def rcv_lfunc_wo(self, contest_id: str, winner: str, loser: str) -> int:
        """
        Check whether vote is a vote for the loser with respect to a 'winner only'
        assertion between the given 'winner' and 'loser'.

        Parameters:
        -----------
        contest_id: string
            identifier for the contest
        winner: string
            identifier for winning candidate
        loser: string
            identifier for losing candidate
        cvr: CVR object

        Returns:
        --------
        1 if the given vote is a vote for 'loser' and 0 otherwise
        """
        rank_winner = self.get_vote_for(contest_id, winner)
        rank_loser = self.get_vote_for(contest_id, loser)

        if not bool(rank_winner) and bool(rank_loser):
            return 1
        elif bool(rank_winner) and bool(rank_loser) and rank_loser < rank_winner:
            return 1
        else:
            return 0

    def rcv_votefor_cand(self, contest_id: str, cand: str, remaining: list) -> int:
        """
        Check whether 'vote' is a vote for the given candidate in the context
        where only candidates in 'remaining' remain standing.

        Parameters:
        -----------
        contest_id: string
            identifier of the contest used in the CVRs
        cand: string
            identifier for candidate
        remaining: list
            list of identifiers of candidates still standing

        vote: dict of dicts

        Returns:
        --------
        1 if the given vote for the contest counts as a vote for 'cand' and 0 otherwise. Essentially,
        if you reduce the ballot down to only those candidates in 'remaining',
        and 'cand' is the first preference, return 1; otherwise return 0.
        """
        if not cand in remaining:
            return 0

        if not bool(rank_cand := self.get_vote_for(contest_id, cand)):
            return 0
        else:
            for altc in remaining:
                if altc == cand:
                    continue
                rank_altc = self.get_vote_for(contest_id, altc)
                if bool(rank_altc) and rank_altc <= rank_cand:
                    return 0
            return 1

    @classmethod
    def cvrs_to_json(cls, cvr):
        return json.dumps(cvr)

    @classmethod
    def from_dict(cls, cvr_dict: list[dict]) -> list:
        """
        Construct a list of CVR objects from a list of dicts containing cvr data

        Parameters:
        -----------
        cvr_dict: a list of dicts, one per cvr

        Returns:
        ---------
        list of CVR objects
        """
        cvr_list = []
        for c in cvr_dict:
            phantom = False if "phantom" not in c.keys() else c["phantom"]
            pool = False if "pool" not in c.keys() else c["pool"]
            tally_pool = None if "tally_pool" not in c.keys() else c["tally_pool"]
            sample_num = None if "sample_num" not in c.keys() else c["sample_num"]
            p = None if "p" not in c.keys() else c["p"]
            sampled = None if "sampled" not in c.keys() else c["sampled"]
            cvr_list.append(
                CVR(
                    id=c["id"],
                    votes=c["votes"],
                    phantom=phantom,
                    pool=pool,
                    tally_pool=tally_pool,
                    sample_num=sample_num,
                    p=p,
                    sampled=sampled,
                )
            )
        return cvr_list

    @classmethod
    def from_raire(cls, raire: list, phantom: bool = False) -> Tuple[list, int]:
        """
        Create a list of CVR objects from a list of cvrs in RAIRE format

        Parameters:
        -----------
        raire: list of comma-separated values
            source in RAIRE format. From the RAIRE documentation:
            The RAIRE format (for later processing) is a CSV file.
            First line: number of contests.
            Next, a line for each contest
             Contest,id,N,C1,C2,C3 ...
                id is the contest_id
                N is the number of candidates in that contest
                and C1, ... are the candidate id's relevant to that contest.
            Then a line for every ranking that appears on a ballot:
             Contest id,Ballot id,R1,R2,R3,...
            where the Ri's are the unique candidate ids.

            The CVR file is assumed to have been read using csv.reader(), so each row has
            been split.

        Returns:
        --------
        list of CVR objects corresponding to the RAIRE cvrs, merged
        number of CVRs read (before merging)
        """
        skip = int(raire[0][0])
        cvr_list = []
        for c in raire[(skip + 1):]:
            contest_id = c[0]
            id = c[1]
            votes = {}
            for j in range(2, len(c)):
                votes[str(c[j])] = j - 1
            cvr_list.append(
                CVR.from_vote(votes, id=id, contest_id=contest_id, phantom=phantom)
            )
        return CVR.merge_cvrs(cvr_list), len(raire) - skip

    @classmethod
    def from_raire_file(cls, cvr_file: str = None) -> Tuple[list, int, int]:
        """
        Read CVR data from a file; construct list of CVR objects from the data

        Parameters
        ----------
        cvr_file : str
            filename

        Returns
        -------
        cvrs: list of CVR objects
        cvrs_read: int
            number of CVRs read
        unique_ids: int
            number of distinct CVR identifiers read
        """
        cvr_in = []
        with open(cvr_file) as f:
            cvr_reader = csv.reader(f, delimiter=",", quotechar='"')
            for row in cvr_reader:
                cvr_in.append(row)
        cvrs, cvrs_read = CVR.from_raire(cvr_in)
        return cvrs, cvrs_read, len(cvrs)

    @classmethod
    def merge_cvrs(cls, cvr_list: list) -> list:
        """
        Takes a list of CVRs that might contain duplicated ballot ids and merges the votes
        so that each identifier is listed only once, and votes from different records for that
        identifier are merged.
        The merge is in the order of the list: if a later mention of a ballot id has votes
        for the same contest as a previous mention, the votes in that contest are updated
        per the later mention.

        If any of the CVRs has phantom==False, sets phantom=False in the result.
        If only one of a multiple has `tally_pool`, set the tally_pool to that value; if they disagree, throw an error.
        Set `pool=True` if any CVR with the ID has `pool=True`


        Parameters:
        -----------
        cvr_list: list of CVRs

        Returns:
        -----------
        list of merged CVRs
        """
        od = OrderedDict()
        for c in cvr_list:
            if c.id not in od:
                od[c.id] = c
            else:
                od[c.id].votes = {**od[c.id].votes, **c.votes}
                od[c.id].phantom = c.phantom and od[c.id].phantom
                od[c.id].pool = c.pool or od[c.id].pool
                if (
                        (od[c.id].tally_pool is None and c.tally_pool is None)
                        or (od[c.id].tally_pool is not None and c.tally_pool is None)
                        or (od[c.id].tally_pool == c.tally_pool)
                ):
                    pass
                elif od[c.id].tally_pool is None and c.tally_pool is not None:
                    od[c.id].tally_pool = c.tally_pool
                else:
                    raise ValueError(
                        f"two CVRs with the same ID have different tally_pools: \n{str(od)=}\n{str(c)=}"
                    )
                od[c.id].pool = od[c.id] or c.pool
        return [v for v in od.values()]

    @classmethod
    def check_tally_pools(cls, cvr_list: Collection['CVR'], force: bool = True) -> list:
        """
        Checks whether every CVR in each tally_pool has the same value of `pool`.
        If `force==True`, set them all to True if any of them is True


        Parameters:
        -----------
        cvr_list: Collection[CVR]
            collection of CVRs to be merged

        force: bool
            set pool equal to the logical union of the pool values for each tally group


        Returns:
        -----------
        list of CVRs.
        """
        od = {}
        for c in cvr_list:
            if c.id not in od:
                od[c.id] = c
            else:
                od[c.id].votes = {**od[c.id].votes, **c.votes}
                od[c.id].phantom = c.phantom and od[c.id].phantom
                od[c.id].pool = c.pool or od[c.id].pool
                if (
                        (od[c.id].tally_pool is None and c.tally_pool is None)
                        or (od[c.id].tally_pool is not None and c.tally_pool is None)
                        or (od[c.id].tally_pool == c.tally_pool)
                ):
                    pass
                elif od[c.id].tally_pool is None and c.tally_pool is not None:
                    oc[c.id].tally_pool = c.tally_pool
                else:
                    raise ValueError(
                        f"two CVRs with the same ID have different tally_pool values: \n{str(od)=}\n{str(c)=}"
                    )
                oc[c.id].pool = oc[c.id] or c.pool
        return [v for v in od.values()]

    @classmethod
    def set_card_in_batch_lex(cls, cvr_list: Collection['CVR'], tally_pool: dict = None) -> dict:
        '''
        For each CVR, set `card_in_batch` to the lexicographic position of its ID within its tally batch.
        Primarily useful to set a canonical ordering of CVRs for ONEAudit when tally batches are physical batches.

        Parameters
        ----------
        cvr_list: list of CVRs
            the CVRs to assign card_in_batch to

        Returns
        -------
        tally_pool_dict: defaultdict
            keys are values of tally_pool; values are sorted lists of CVR IDs in that tally_pool

        Side Effects
        ------------
        Set `card_in_batch` to the lexicographic position of the CVR ID within its tally_batch
        '''
        tally_pool_dict = defaultdict(list)
        for c in cvr_list:
            tally_pool_dict[c.tally_pool].append(c.id)
        for tp, id_list in tally_pool_dict.items():
            id_list.sort()
        for c in cvr_list:
            c.card_in_batch = tally_pool_dict[c.tally_pool].index(c.id)
        return tally_pool_dict

    @classmethod
    def from_vote(
            cls, vote: str, id: object = 1, contest_id: str = "AvB", phantom: bool = False
    ) -> dict:
        """
        Wraps a vote and creates a CVR, for unit tests

        Parameters:
        ----------
        vote: dict of votes in one contest
        id: str
            CVR id
        contest_id: str
            identifier of the contest

        Returns:
        --------
        CVR containing that vote in the contest "AvB", with CVR id=1.
        """
        return CVR(id=id, votes={contest_id: vote}, phantom=phantom)

    @classmethod
    def as_vote(cls, v) -> int:
        return int(bool(v))

    @classmethod
    def as_rank(cls, v) -> int:
        return int(v)

    @classmethod
    def pool_contests(cls, cvrs: list['CVR']) -> dict:
        """
        return a dict containing, for each tally_pool that is pooled, a list of all contest ids on any CVR in that pool.


        Parameters
        ----------
        cvrs: list of CVR objects
            the set to collect contests from

        Returns
        -------
        dict: keys are tally_pool with pool==True, values are the set of contests mentioned on any CVR in that tally_pool
        """
        tally_pools = defaultdict(set)
        for c in cvrs:
            if c.pool:
                tally_pools[c.tally_pool] = tally_pools[c.tally_pool].union(set(c.votes.keys()))
        return tally_pools

    @classmethod
    def add_pool_contests(cls, cvrs: list['CVR'], tally_pools: dict) -> bool:
        """
        for each tally_pool, ensure every CVR in that tally_pool has every contest mentioned in that pool

        Parameters
        ----------
        cvrs : list of CVR objects
            the set to update with additional contests as needed

        tally_pools dict
            keys are tally_pool ids, values are sets of contests every CVR in that pool should have

        Returns
        -------
        bool : True if any contest is added to any CVR
        """
        added = False
        for c in [d for d in cvrs if (d.tally_pool in tally_pools.keys() and d.pool)]:
            added = (
                    c.update_votes({con: {} for con in tally_pools[c.tally_pool]}) or added
            )  # note: order of terms matters!
        return added

    @classmethod
    def make_phantoms(
            cls,
            audit: dict = None,
            contests: dict = None,
            cvr_list: list['CVR'] = None,
            prefix: str = "phantom-",
            tally_pool=None,
            pool=False

    ) -> Tuple[list, int]:
        """
        Make phantom CVRs as needed for phantom cards; set contest parameters `cards` (if not set) and `cvrs`

        **Currently only works for unstratified audits.**
        If `audit.strata[s]['use_style']`, phantoms are "per contest": each contest needs enough to account for the
        difference between the number of cards that might contain the contest and the number of CVRs that contain
        the contest. This can result in having more cards in all (manifest and phantoms) than max_cards, the maximum cast.

        If `not use_style`, phantoms are for the election as a whole: need enough to account for the difference
        between the number of cards in the manifest and the number of CVRs that contain the contest. Then, the total
        number of cards (manifest plus phantoms) equals max_cards.

        If `not use_style` sets `cards = max_cards` for each contest

        Parameters
        ----------
        cvr_list: list of CVR objects
            the reported CVRs
        contests: dict of contests
            information about each contest under audit
        prefix: String
            prefix for ids for phantom CVRs to be added
        tally_pool: object
            label for tally_pool for pooled CVRs for ONEAudit
        pool: bool
            pool the votes for the CVRs?


        Returns
        -------
        cvr_list: list of CVR objects
            the reported CVRs and the phantom CVRs
        n_phantoms: int
            number of phantom cards added

        Side effects
        ------------
        for each contest in `contests`, sets `cards` to max_cards if not specified by the user or if `not use_style`
        for each contest in `contests`, set `cvrs` to be the number of (real) CVRs that contain the contest
        """
        if len(audit.strata) > 1:
            raise NotImplementedError("stratified audits not implemented")
        stratum = next(iter(audit.strata.values()))
        use_style = stratum.use_style
        max_cards = stratum.max_cards
        phantom_vrs = []
        n_cvrs = len(cvr_list)
        for c, con in contests.items():  # set contest parameters
            con.cvrs = np.sum(
                [cvr.has_contest(con.id) for cvr in cvr_list if not cvr.phantom]
            )
            con.cards = (
                max_cards if ((con.cards is None) or (not use_style)) else con.cards
            )
        # Note: this will need to change for stratified audits
        if not use_style:  # make (max_cards - len(cvr_list)) phantoms
            phantoms = max_cards - n_cvrs
            for i in range(phantoms):
                phantom_vrs.append(
                    CVR(id=prefix + str(i + 1), votes={}, phantom=True, tally_pool=tally_pool, pool=pool))
        else:  # create phantom CVRs as needed for each contest
            for c, con in contests.items():
                phantoms_needed = con.cards - con.cvrs
                while len(phantom_vrs) < phantoms_needed:  # creat additional phantoms
                    phantom_vrs.append(
                        CVR(
                            id=prefix + str(len(phantom_vrs) + 1),
                            votes={},
                            phantom=True,
                            tally_pool=tally_pool,
                            pool=pool
                        )
                    )
                for i in range(phantoms_needed):
                    phantom_vrs[i].votes[
                        con.id
                    ] = {}  # list contest c on the phantom CVR
            phantoms = len(phantom_vrs)
        cvr_list = cvr_list + phantom_vrs
        return cvr_list, phantoms

    @classmethod
    def assign_sample_nums(cls, cvr_list: list['CVR'], prng: 'np.RandomState') -> bool:
        """
        Assigns a pseudo-random sample number to each cvr in cvr_list

        Parameters
        ----------
        cvr_list: list of CVR objects
        prng: instance of cryptorandom SHA256 generator

        Returns
        -------
        True

        Side effects
        ------------
        assigns (or overwrites) sample numbers in each CVR in cvr_list
        """
        for cvr in cvr_list:
            cvr.sample_num = int_from_hash(prng.nextRandom())
        return True

    @classmethod
    def prep_comparison_sample(
            cls, mvr_sample: list['CVR'], cvr_sample: list['CVR'], sample_order: list
    ):
        """
        prepare the MVRs and CVRs for comparison by putting them into the same (random) order
        in which the CVRs were selected

        conduct data integrity checks.

        Side-effects: sorts the mvr sample into the same order as the cvr sample

        Parameters
        ----------
        mvr_sample: list of CVR objects
            the manually determined votes for the audited cards
        cvr_sample: list of CVR objects
            the electronic vote record for the audited cards
        sample_order: dict
            dict to look up selection order of the cards. Keys are card ids. Values are dicts
            containing "selection_order" (which draw yielded the card) and "serial" (the card's original position)

        Returns
        -------

        Side effects
        ------------
        sorts the mvr sample into the same order as the cvr sample
        """
        mvr_sample.sort(key=lambda x: sample_order[x.id]["selection_order"])
        cvr_sample.sort(key=lambda x: sample_order[x.id]["selection_order"])
        assert len(cvr_sample) == len(
            mvr_sample
        ), "Number of cvrs ({}) and number of mvrs ({}) differ".format(
            len(cvr_sample), len(mvr_sample)
        )
        for i in range(len(cvr_sample)):
            assert (
                    mvr_sample[i].id == cvr_sample[i].id
            ), f"Mismatch between id of cvr ({cvr_sample[i].id}) and mvr ({mvr_sample[i].id})"

    @classmethod
    def prep_polling_sample(cls, mvr_sample: list, sample_order: dict):
        """
        Put the mvr sample back into the random selection order.

        Only about the side effects.

        Parameters
        ----------
        mvr_sample: list
            list of CVR objects
        sample_order: dict of dicts
            dict to look up selection order of the cards. Keys are card identifiers. Values are dicts
            containing "selection_order" (which draw yielded the card) and "serial" (the card's original position)

        Returns
        -------

        Side Effects
        -------------
        mvr_sample is reordered into the random selection order
        """
        mvr_sample.sort(key=lambda x: sample_order[x.id]["selection_order"])

    @classmethod
    def sort_cvr_sample_num(cls, cvr_list: list):
        """
        Sort cvr_list by sample_num

        Only about the side effects.

        Parameters
        ----------
        cvr_list: list
            list of CVR objects

        Returns
        -------
        True

        Side effects
        ------------
        cvr_list is sorted by sample_num
        """
        cvr_list.sort(key=lambda x: x.sample_num)
        return True

    @classmethod
    def consistent_sampling(
            cls,
            cvr_list: list['CVR'] = None,
            contests: dict = None,
            sampled_cvr_indices: list = None,
    ) -> list:
        """
        Sample CVR ids for contests to attain sample sizes in contests, a dict of Contest objects

        Assumes that phantoms have already been generated and sample_num has been assigned
        to every CVR, including phantoms

        Parameters
        ----------
        cvr_list: list
            list of CVR objects
        contests: dict
            dict of Contest objects. Contest sample sizes must be set before calling this function.
        sampled_cvr_indices: list
            indices of cvrs already in the sample

        Returns
        -------
        sampled_cvr_indices: list
            indices of CVRs to sample (0-indexed)
        """
        current_sizes = defaultdict(int)
        contest_in_progress = lambda c: (current_sizes[c.id] < c.sample_size)
        if sampled_cvr_indices is None:
            sampled_cvr_indices = []
        else:
            for sam in sampled_cvr_indices:
                for c, con in contests.items():
                    current_sizes[c] += 1 if cvr_list[sam].has_contest(con.id) else 0
        sorted_cvr_indices = [
            i for i, cv in sorted(enumerate(cvr_list), key=lambda x: x[1].sample_num)
        ]
        inx = len(sampled_cvr_indices)
        while any([contest_in_progress(con) for c, con in contests.items()]):
            if any(
                    [
                        (
                                contest_in_progress(con)
                                and cvr_list[sorted_cvr_indices[inx]].has_contest(con.id)
                        )
                        for c, con in contests.items()
                    ]
            ):
                sampled_cvr_indices.append(sorted_cvr_indices[inx])
                for c, con in contests.items():
                    if cvr_list[sorted_cvr_indices[inx]].has_contest(
                            con.id
                    ) and contest_in_progress(con):
                        con.sample_threshold = cvr_list[
                            sorted_cvr_indices[inx]
                        ].sample_num
                        current_sizes[c] += 1
            inx += 1
        for i in sampled_cvr_indices:
            cvr_list[i].sampled = True
        return sampled_cvr_indices

    @classmethod
    def tabulate_styles(cls, cvr_list: Collection['CVR'] = None):
        """
        tabulate unique CVR styles in cvr_list

        Parameters
        ----------
        cvr_list: Collection
            collection of CVR objects

        Returns
        -------
        a dict of styles and the counts of those styles
        """
        # iterate through and find all the unique styles
        style_counts = defaultdict(int)
        for cvr in cvr_list:
            style_counts[frozenset(cvr.votes.keys())] += 1
        return style_counts

    @classmethod
    def tabulate_votes(cls, cvr_list: Collection['CVR'] = None):
        """
        tabulate total votes for each candidate in each contest in cvr_list.
        For plurality, supermajority, and approval. Not useful for ranked-choice voting.

        Parameters
        ----------
        cvr_list: Collection
            collection of CVR objects

        Returns
        -------
        dict of dicts:
            main key is contest
            sub key is the candidate in the contest
            value is the number of votes for that candidate in that contest
        """
        d = defaultdict(lambda: defaultdict(int))
        for c in cvr_list:
            for con, votes in c.votes.items():
                for cand in votes:
                    d[con][cand] += CVR.as_vote(c.get_vote_for(con, cand))
        return d

    @classmethod
    def tabulate_cards_contests(cls, cvr_list: Collection = None):
        """
        Tabulate the number of cards containing each contest

        Parameters
        ----------
        cvr_list: Collection
            collection of CVR objects

        Returns
        -------
        dict:
            main key is contest
            value is the number of cards containing that contest
        """
        d = defaultdict(int)
        for c in cvr_list:
            for con in c.votes:
                d[con] += 1
        return d


# @dataclass
# class Tabulation:
#     pass
#     # tabulation: list[dict[str, int]] = None
#     #
#     # def get_winners(self, n_winners: int = 1) -> list[str]:
#     #     """
#     #     Get the winners from the tabulation.
#     #     """
#     #     if self.tabulation is None:
#     #         raise ValueError("No tabulation available.")
#     #     winners = []
#     #     for t in self.tabulation:
#     #         if len(winners) < n_winners:
#     #             winners.append(t["candidate"])
#     #         else:
#     #             break
#     #     return winners


@dataclass
class SocialChoiceFunction(ABC):
    n_winners: int = 1
    winners: set[str] = None

    # @abstractmethod
    # def tabulate(self, votes: Collection[CVR], candidates: list[str]) -> Tabulation:
    #     """
    #     Tabulate the votes according to the social choice function.
    #     """
    #     pass

    def assort_upper_bound(self):
        return 1

    @abstractmethod
    def assort(self,
               c: CVR,
               contest_id,
               winner: str,
               loser: str,
               winner_cands: list[str],
               loser_cands: list[str] = None
               ) -> float:
        pass

    # @abstractmethod
    # def get_winner_loser_pairs(self,
    #                            winners: list[str],
    #                            losers: list[str]
    #                            ) -> Iterable[tuple[str, str, str, list[str], list[str]]]:
    #     """
    #     Get the winner-loser pairs for the social choice function.
    #
    #     Parameters
    #     ----------
    #     winners: list[str]
    #         List of winners.
    #     losers: list[str]
    #         List of losers.
    #
    #     Returns
    #     -------
    #     Iterable[tuple[str, str, str, list[str], list[str]]]:
    #         Iterable of tuples containing the name, winner, loser, winner_candidates, loser_candidates.
    #         winner_/loser_candidates mean the set of candidates under consideration when evaluating the winner/loser
    #     """
    #     pass


class Plurality(SocialChoiceFunction):
    def assort(self,
               c: CVR,
               contest_id,
               winner: str,
               loser: str,
               winner_cands: list[str],
               loser_cands: list[str] = None
               ) -> float:
        if loser_cands is None:
            loser_cands = winner_cands
        winr = CVR.as_vote(c.get_vote_for(contest_id, winner)) if winner in winner_cands else 0
        losr = CVR.as_vote(c.get_vote_for(contest_id, loser)) if loser in loser_cands else 0
        return (winr - losr + 1) / 2

    def get_winner_loser_pairs(self,
                              winners: list[str],
                              losers: list[str]
                              ) -> Iterable[tuple[str, str, list[str], list[str]]]:
        cands = winners + losers
        for winner in winners:
            for loser in losers:
                yield f"{winner} v {loser}", winner, loser, cands, cands


@dataclass
class Threshold(SocialChoiceFunction):
    threshold: float = 2/3

    def assort_upper_bound(self):
        return 1 / (2 * self.threshold)

    def assort(self, c: CVR, contest_id, winner: str, loser: str,
               winner_cands: list[str], loser_cands: list[str] = None) -> float:
        winr = CVR.as_vote(c.get_vote_for(contest_id, winner))
        return winr / (2 * self.threshold) if c.has_one_vote(contest_id, winner_cands) else 1 / 2

    def get_winner_loser_pairs(self,
                              winners: list[str],
                              losers: list[str]
                              ) -> Iterable[tuple[str, str, str, list[str], list[str]]]:
        cands = winners + losers
        for winner in winners:
            for loser in losers:
                yield winner, loser, cands, cands


Supermajority = Threshold


class Approval(SocialChoiceFunction):
    pass


class DHont(SocialChoiceFunction):
    pass


@dataclass
class InstantRunoff(SocialChoiceFunction):
    max_vote_length: int = np.inf  # TODO: use this to truncate/eliminate CVRs that exceeds max length?

    def assort(self, c: CVR, contest_id, winner: str, loser: str,
               winner_cands: list[str], loser_cands: list[str] = None) -> float:
        if loser_cands is None:
            loser_cands = winner_cands

        winr = c.rcv_votefor_cand(contest_id, winner, winner_cands)
        losr = c.rcv_votefor_cand(contest_id, loser, loser_cands)

        return (winr - losr + 1) / 2


class Contest:
    """
    Objects and methods for contests.
    """

    class CANDIDATES:
        """
        constants for referring to candidates and candidate groups.

        For example, in a supermajority contest where no candidate is reported to have won,
        the winner is Contest.CANDIDATES.NO_CANDIDATE, and in a supermajority contest in which one
        candidate is reported to have won, the loser (for the assorter) is Contest.CANDIDATES.ALL_OTHERS
        """

        CANDIDATES = (
            ALL := "ALL",
            ALL_OTHERS := "ALL_OTHERS",
            WRITE_IN := "WRITE_IN",
            NO_CANDIDATE := "NO_CANDIDATE",
        )

    ATTRIBUTES = (
        "id",
        "name",
        "risk_limit",
        "cards",
        "choice_function",
        "n_winners",
        "share_to_win",
        "candidates",
        "winner",
        "assertion_file",
        "audit_type",
        "test",
        "test_kwargs",
        "g",
        "use_style",
        "assertions",
        "tally",
        "sample_size",
        "sample_threshold",
    )

    def __init__(
            self,
            id: object = None,
            name: str = None,
            risk_limit: float = 0.05,
            cards: int = 0,
            choice_function: SocialChoiceFunction = Plurality,
            n_winners: int = 1,
            share_to_win: float = None,
            candidates: Collection = None,
            winner: Collection = None,
            assertion_file: str = None,
            audit_type: str = Audit.AUDIT_TYPE.CARD_COMPARISON,
            test: callable = None,
            test_kwargs: dict = {},
            g: float = 0.1,
            estim: callable = None,
            bet: callable = None,
            use_style: bool = True,
            assertions: dict = None,
            tally: dict = None,
            sample_size: int = None,
            sample_threshold: float = None,
    ):
        self.id = id
        self.name = name
        self.risk_limit = risk_limit
        self.cards = cards
        self.choice_function = choice_function
        self.n_winners = n_winners
        self.share_to_win = share_to_win
        self.candidates = candidates
        self.winner = winner
        self.assertion_file = assertion_file
        self.audit_type = audit_type
        self.test = test
        self.test_kwargs = test_kwargs
        self.g = g
        self.estim = estim
        self.bet = bet
        self.use_style = use_style
        self.assertions = assertions
        self.tally = tally
        self.sample_size = sample_size
        self.sample_threshold = sample_threshold

    def __str__(self):
        return str(self.__dict__)

    def find_sample_size(
            self,
            audit: object = None,
            mvr_sample: list = None,
            cvr_sample: Collection = None,
            **kwargs,
    ) -> int:
        """
        Estimate the sample size required to confirm the contest at its risk limit.

        This function can be used with or without data, for Audit.AUDIT_TYPE.POLLING,
        Audit.AUDIT_TYPE.CARD_COMPARISON, and Audit.AUDIT_TYPE.ONEAUDIT audits.

        The simulations in this implementation are inefficient because the randomization happens separately
        for every assorter, rather than in parallel.

        Parameters
        ----------
        cvrs: list of CVRs
            data (or simulated data) to base the sample size estimates on
        mvrs: list of MVRs (CVR objects)
            manually read votes to base the sample size estimates on, if data are available.

        Returns
        -------
        estimated sample size

        Side effects
        ------------
        sets self.sample_size to the estimated sample size

        """
        self.sample_size = 0
        for a in self.assertions.values():
            data = None
            if mvr_sample is not None:  # process the MVRs/CVRs to get data appropriate to each assertion
                data, u = a.mvrs_to_data(mvr_sample, cvr_sample)
            elif self.audit_type == Audit.AUDIT_TYPE.ONEAUDIT:
                data, u = a.mvrs_to_data(cvr_sample, cvr_sample)
            self.sample_size = max(
                self.sample_size,
                a.find_sample_size(
                    data=data,
                    rate_1=audit.error_rate_1,
                    rate_2=audit.error_rate_2,
                    reps=audit.reps,
                    quantile=audit.quantile,
                    seed=audit.sim_seed,
                ),
            )
        return self.sample_size

    def find_margins_from_tally(self):
        """
        Use the `Contest.tally` attribute to set the margins of the contest's assorters.

        Appropriate only for the social choice functions
                Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY,
                Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY,
                Contest.SOCIAL_CHOICE_FUNCTION.APPROVAL


        Parameters
        ----------
        None

        Returns
        -------
        None

        Side effects
        ------------
        sets Assertion.margin for all Assertions in the Contest
        """
        for a, assn in self.assertions.items():
            assn.find_margin_from_tally()

    def make_assort_function(self, winner: str, loser: str | None,
                             winner_cands: list[str] = None, loser_cands: list[str] = None) -> Callable[[CVR], float]:
        if winner_cands is None:
            winner_cands = list(self.candidates)
        return lambda v: self.choice_function.assort(v, contest_id=self.id, winner=winner, loser=loser,
                                                       winner_cands=winner_cands, loser_cands=loser_cands)

    @classmethod
    def check_cards(cls, contests: Collection['Contest'], cvrs: Collection[CVR], force: bool = False):
        '''
        Check whether the number of CVRs that contain each contest is not greater than the upper bound
        on the number of cards that contain the contest; optionally, increase the upper bounds to make that so.

        Parameters
        ----------
        contests: collection of Contests

        cvrs: collection of CVRs

        force: bool
            Increase the upper bounds to include all the CVRs.
            This is useful for ONEAudit when the original upper bounds were fine but ONEAudit added the contest
            to some CVRs in some pool batches.
        '''
        for c, con in contests.items():
            found = np.sum([cvr.has_contest(c) for cvr in cvrs])
            if found > con.cards:
                if not force:
                    raise ValueError(f'{found} cards contain contest {c} but upper bound is {con.cards}')
                else:
                    warnings.warn(f'{found} cards contain contest {c} but upper bound is {con.cards}')
            con.cards = max(con.cards, found) if force else con.cards

    @classmethod
    def tally(cls, con_dict: dict = None, cvr_list: Collection[CVR] = None, enforce_rules: bool = True) -> dict:
        """
        Tally the votes in the contests in con_dict from a collection of CVRs.
        Only tallies plurality, multi-winner plurality, supermajority, and approval contests.

        Parameters
        ----------
        con_dict: dict
            dict of Contest objects to find tallies for
        cvr_list: list[CVR]
            list of CVRs containing the votes to tally
        enforce_rules: bool
            Enforce the voting rules for the social choice function?
            For instance, if the contest is a vote-for-k plurality and the CVR has more than k votes,
            then if `enforce_rules`, no candidate's total is incremented, but if `not enforce_rules`,
            the tally for every candidate with a vote is incremented.

        Returns
        -------

        Side Effects
        ------------
        Sets the `tally` dict for the contests in con_list, if their social choice function is appropriate
        """
        tallies = {}
        cons = []
        for id, c in con_dict.items():
            if type(c.choice_function) in [Plurality, Threshold, Approval]:
                cons.append(c)
                c.tally = defaultdict(int)
            else:
                warnings.warn(
                    f"contest {c.id} ({c.name}) has social choice function "
                    + f"{c.choice_function}: not tabulated"
                )
        for cvr in cvr_list:
            for c in cons:
                if cvr.has_contest(c.id):
                    if enforce_rules:
                        n_votes = 0
                        for candidate, vote in cvr.votes[c.id].items():
                            if candidate:
                                n_votes += int(bool(vote))
                    if (not enforce_rules) or (n_votes <= c.n_winners):
                        for candidate, vote in cvr.votes[c.id].items():
                            if candidate:
                                c.tally[candidate] += int(bool(vote))

    @classmethod
    def from_dict(cls, d: dict) -> dict:
        """
        define a contest objects from a dict containing data for one contest
        """
        c = Contest()
        c.__dict__.update(d)
        return c

    @classmethod
    def from_dict_of_dicts(cls, d: dict) -> dict:
        """
        define a dict of contest objects from a dict of dicts, each inner dict containing data for one contest
        """
        contests = {}
        for di, v in d.items():
            contests[di] = cls.from_dict(v)
            contests[di].id = di
        return contests

    @classmethod
    def from_cvr_list(
            cls, audit, votes, cards, cvr_list: Collection[CVR] = None
    ) -> dict:
        """
        Create a contest dict containing all contests in cvr_list.
        Every contest is single-winner plurality by default, audited by ballot comparison
        """
        if len(audit.strata) > 1:
            raise NotImplementedError("stratified audits not implemented")
        stratum = next(iter(audit.strata.values()))
        use_style = stratum.use_style
        max_cards = stratum.max_cards
        contest_dict = {}
        for key in votes:
            contest_name = str(key)
            cards_with_contest = cards[key]
            options = np.array(list(votes[key].keys()), dtype="str")
            tallies = np.array(list(votes[key].values()))

            reported_winner = options[np.argmax(tallies)]

            contest_dict[contest_name] = {
                "name": contest_name,
                "cards": cards_with_contest if use_style else max_cards,
                "choice_function": Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY,
                "n_winners": 1,
                "risk_limit": 0.05,
                "candidates": list(options),
                "winner": [reported_winner],
                "assertion_file": None,
                "audit_type": Audit.AUDIT_TYPE.CARD_COMPARISON,
                "test": NonnegMean.alpha_mart,
                "estim": NonnegMean.optimal_comparison,
                "bet": NonnegMean.fixed_bet,
            }
        contests = Contest.from_dict_of_dicts(contest_dict)
        return contests

    @classmethod
    def print_margins(cls, contests: dict = None):
        """
        print all assorter margins
        """
        for c, con in contests.items():
            print(f"margins in contest {c}:")
            for a, m in con.margins.items():
                print(f"\tassertion {a}: {m}")
