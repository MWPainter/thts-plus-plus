#include "../search/search.h"

#include "../search/searchnode.h"

#include <limits>
#include <cmath>
#include <iostream>
const double CONST_E = std::exp(1.0);

//------------------------
#include "../core/using.h"
//------------------------

static double cpuctExploration(double totalChildWeight, const SearchParams& searchParams) {
  return searchParams.cpuctExploration +
    searchParams.cpuctExplorationLog * log((totalChildWeight + searchParams.cpuctExplorationBase) / searchParams.cpuctExplorationBase);
}

//Tiny constant to add to numerator of puct formula to make it positive
//even when visits = 0.
static constexpr double TOTALCHILDWEIGHT_PUCT_OFFSET = 0.01;

double Search::getExploreSelectionValue(
  double nnPolicyProb, double totalChildWeight, double childWeight,
  double childUtility, double parentUtilityStdevFactor, Player pla
) const {
  if(nnPolicyProb < 0)
    return POLICY_ILLEGAL_SELECTION_VALUE;

  double exploreComponent = 0.0;
  if (using_bts) {
    exploreComponent = cpuctExploration(totalChildWeight,searchParams)
      * parentUtilityStdevFactor
      * nnPolicyProb
      * sqrt(totalChildWeight + TOTALCHILDWEIGHT_PUCT_OFFSET)
      / (1.0 + childWeight);
  }

  //At the last moment, adjust value to be from the player's perspective, so that players prefer values in their favor
  //rather than in white's favor
  double valueComponent = pla == P_WHITE ? childUtility : -childUtility;
  return exploreComponent + valueComponent;
}

//Return the childWeight that would make Search::getExploreSelectionValue return the given explore selection value.
//Or return 0, if it would be less than 0.
double Search::getExploreSelectionValueInverse(
  double exploreSelectionValue, double nnPolicyProb, double totalChildWeight,
  double childUtility, double parentUtilityStdevFactor, Player pla
) const {
  if(nnPolicyProb < 0)
    return 0;
  double valueComponent = pla == P_WHITE ? childUtility : -childUtility;

  double exploreComponent = exploreSelectionValue - valueComponent;
  double exploreComponentScaling =
    cpuctExploration(totalChildWeight,searchParams)
    * parentUtilityStdevFactor
    * nnPolicyProb
    * sqrt(totalChildWeight + TOTALCHILDWEIGHT_PUCT_OFFSET);

  //Guard against float weirdness
  if(exploreComponent <= 0)
    return 1e100;

  double childWeight = exploreComponentScaling / exploreComponent - 1;
  if(childWeight < 0)
    childWeight = 0;
  return childWeight;
}

static void maybeApplyWideRootNoise(
  double& childUtility,
  float& nnPolicyProb,
  const SearchParams& searchParams,
  SearchThread* thread,
  const SearchNode& parent
) {
  //For very large wideRootNoise, go ahead and also smooth out the policy
  nnPolicyProb = (float)pow(nnPolicyProb, 1.0 / (4.0*searchParams.wideRootNoise + 1.0));
  if(thread->rand.nextBool(0.5)) {
    double bonus = searchParams.wideRootNoise * std::fabs(thread->rand.nextGaussian());
    if(parent.nextPla == P_WHITE)
      childUtility += bonus;
    else
      childUtility -= bonus;
  }
}


double Search::getExploreSelectionValueOfChild(
  const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
  Loc moveLoc,
  double totalChildWeight, int64_t childEdgeVisits, double fpuValue,
  double parentUtility, double parentWeightPerVisit, double parentUtilityStdevFactor,
  bool isDuringSearch, bool antiMirror, double maxChildWeight, SearchThread* thread
) const {
  (void)parentUtility;
  int movePos = getPos(moveLoc);
  float nnPolicyProb = parentPolicyProbs[movePos];

  int32_t childVirtualLosses = child->virtualLosses.load(std::memory_order_acquire);
  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  double utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);
  double scoreMeanAvg = child->stats.scoreMeanAvg.load(std::memory_order_acquire);
  double scoreMeanSqAvg = child->stats.scoreMeanSqAvg.load(std::memory_order_acquire);
  double childWeight = child->stats.getChildWeight(childEdgeVisits,childVisits);

  //It's possible that childVisits is actually 0 here with multithreading because we're visiting this node while a child has
  //been expanded but its thread not yet finished its first visit.
  //It's also possible that we observe childWeight <= 0 even though childVisits >= due to multithreading, the two could
  //be out of sync briefly since they are separate atomics.
  double childUtility;
  if(childVisits <= 0 || childWeight <= 0.0)
    childUtility = fpuValue;
  else {
    childUtility = utilityAvg;

    //Tiny adjustment for passing
    double endingScoreBonus = getEndingWhiteScoreBonus(parent,moveLoc);
    if(endingScoreBonus != 0)
      childUtility += getScoreUtilityDiff(scoreMeanAvg, scoreMeanSqAvg, endingScoreBonus);
  }

  //When multithreading, totalChildWeight could be out of sync with childWeight, so if they provably are, then fix that up
  if(totalChildWeight < childWeight)
    totalChildWeight = childWeight;

  //Virtual losses to direct threads down different paths
  if(childVirtualLosses > 0) {
    double virtualLossWeight = childVirtualLosses * searchParams.numVirtualLossesPerThread;

    double utilityRadius = searchParams.winLossUtilityFactor + searchParams.staticScoreUtilityFactor + searchParams.dynamicScoreUtilityFactor;
    double virtualLossUtility = (parent.nextPla == P_WHITE ? -utilityRadius : utilityRadius);
    double virtualLossWeightFrac = (double)virtualLossWeight / (virtualLossWeight + std::max(0.25,childWeight));
    childUtility = childUtility + (virtualLossUtility - childUtility) * virtualLossWeightFrac;
    childWeight += virtualLossWeight;
  }

  if(isDuringSearch && (&parent == rootNode)) {
    //Futile visits pruning - skip this move if the amount of time we have left to search is too small, assuming
    //its average weight per visit is maintained.
    //We use childVisits rather than childEdgeVisits for the final estimate since when childEdgeVisits < childVisits, adding new visits is instant.
    if(searchParams.futileVisitsThreshold > 0) {
      double requiredWeight = searchParams.futileVisitsThreshold * maxChildWeight;
      //Avoid divide by 0 by adding a prior equal to the parent's weight per visit
      double averageVisitsPerWeight = (childEdgeVisits + 1.0) / (childWeight + parentWeightPerVisit);
      double estimatedRequiredVisits = requiredWeight * averageVisitsPerWeight;
      if(childVisits + thread->upperBoundVisitsLeft < estimatedRequiredVisits)
        return FUTILE_VISITS_PRUNE_VALUE;
    }
    //Hack to get the root to funnel more visits down child branches
    if(searchParams.rootDesiredPerChildVisitsCoeff > 0.0) {
      if(childWeight < sqrt(nnPolicyProb * totalChildWeight * searchParams.rootDesiredPerChildVisitsCoeff)) {
        return 1e20;
      }
    }
    //Hack for hintloc - must search this move almost as often as the most searched move
    if(rootHintLoc != Board::NULL_LOC && moveLoc == rootHintLoc) {
      double averageWeightPerVisit = (childWeight + parentWeightPerVisit) / (childVisits + 1.0);
      int childrenCapacity;
      const SearchChildPointer* children = parent.getChildren(childrenCapacity);
      for(int i = 0; i<childrenCapacity; i++) {
        const SearchNode* c = children[i].getIfAllocated();
        if(c == NULL)
          break;
        int64_t cEdgeVisits = children[i].getEdgeVisits();
        double cWeight = c->stats.getChildWeight(cEdgeVisits);
        if(childWeight + averageWeightPerVisit < cWeight * 0.8)
          return 1e20;
      }
    }

    if(searchParams.wideRootNoise > 0.0) {
      maybeApplyWideRootNoise(childUtility, nnPolicyProb, searchParams, thread, parent);
    }
  }
  if(isDuringSearch && antiMirror) {
    maybeApplyAntiMirrorPolicy(nnPolicyProb, moveLoc, parentPolicyProbs, parent.nextPla, thread);
    maybeApplyAntiMirrorForcedExplore(childUtility, parentUtility, moveLoc, parentPolicyProbs, childWeight, totalChildWeight, parent.nextPla, thread, parent);
  }

  return getExploreSelectionValue(nnPolicyProb,totalChildWeight,childWeight,childUtility,parentUtilityStdevFactor,parent.nextPla);
}

double Search::getNewExploreSelectionValue(
  const SearchNode& parent, float nnPolicyProb,
  double totalChildWeight, double fpuValue,
  double parentWeightPerVisit, double parentUtilityStdevFactor,
  double maxChildWeight, SearchThread* thread
) const {
  double childWeight = 0;
  double childUtility = fpuValue;
  if(&parent == rootNode) {
    //Futile visits pruning - skip this move if the amount of time we have left to search is too small
    if(searchParams.futileVisitsThreshold > 0) {
      //Avoid divide by 0 by adding a prior equal to the parent's weight per visit
      double averageVisitsPerWeight = 1.0 / parentWeightPerVisit;
      double requiredWeight = searchParams.futileVisitsThreshold * maxChildWeight;
      double estimatedRequiredVisits = requiredWeight * averageVisitsPerWeight;
      if(thread->upperBoundVisitsLeft < estimatedRequiredVisits)
        return FUTILE_VISITS_PRUNE_VALUE;
    }
    if(searchParams.wideRootNoise > 0.0) {
      maybeApplyWideRootNoise(childUtility, nnPolicyProb, searchParams, thread, parent);
    }
  }
  return getExploreSelectionValue(nnPolicyProb,totalChildWeight,childWeight,childUtility,parentUtilityStdevFactor,parent.nextPla);
}

double Search::getReducedPlaySelectionWeight(
  const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
  Loc moveLoc,
  double totalChildWeight, int64_t childEdgeVisits,
  double parentUtilityStdevFactor, double bestChildExploreSelectionValue
) const {
  assert(&parent == rootNode);
  int movePos = getPos(moveLoc);
  float nnPolicyProb = parentPolicyProbs[movePos];

  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  double scoreMeanAvg = child->stats.scoreMeanAvg.load(std::memory_order_acquire);
  double scoreMeanSqAvg = child->stats.scoreMeanSqAvg.load(std::memory_order_acquire);
  double utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);
  double childWeight = child->stats.getChildWeight(childEdgeVisits,childVisits);

  //Child visits may be 0 if this function is called in a multithreaded context, such as during live analysis
  //Child weight may also be 0 if it's out of sync.
  if(childVisits <= 0 || childWeight <= 0.0)
    return 0;

  //Tiny adjustment for passing
  double endingScoreBonus = getEndingWhiteScoreBonus(parent,moveLoc);
  double childUtility = utilityAvg;
  if(endingScoreBonus != 0)
    childUtility += getScoreUtilityDiff(scoreMeanAvg, scoreMeanSqAvg, endingScoreBonus);

  double childWeightWeRetrospectivelyWanted = getExploreSelectionValueInverse(
    bestChildExploreSelectionValue, nnPolicyProb, totalChildWeight, childUtility, parentUtilityStdevFactor, parent.nextPla
  );
  if(childWeight > childWeightWeRetrospectivelyWanted)
    return childWeightWeRetrospectivelyWanted;
  return childWeight;
}

double Search::getFpuValueForChildrenAssumeVisited(
  const SearchNode& node, Player pla, bool isRoot, double policyProbMassVisited,
  double& parentUtility, double& parentWeightPerVisit, double& parentUtilityStdevFactor
) const {
  int64_t visits = node.stats.visits.load(std::memory_order_acquire);
  double weightSum = node.stats.weightSum.load(std::memory_order_acquire);
  double utilityAvg = node.stats.utilityAvg.load(std::memory_order_acquire);
  double utilitySqAvg = node.stats.utilitySqAvg.load(std::memory_order_acquire);

  assert(visits > 0);
  assert(weightSum > 0.0);
  parentWeightPerVisit = weightSum / visits;
  parentUtility = utilityAvg;
  double variancePrior = searchParams.cpuctUtilityStdevPrior * searchParams.cpuctUtilityStdevPrior;
  double variancePriorWeight = searchParams.cpuctUtilityStdevPriorWeight;
  double parentUtilityStdev;
  if(visits <= 0 || weightSum <= 1)
    parentUtilityStdev = searchParams.cpuctUtilityStdevPrior;
  else {
    double utilitySq = parentUtility * parentUtility;
    //Make sure we're robust to numerical precision issues or threading desync of these values, so we don't observe negative variance
    if(utilitySqAvg < utilitySq)
      utilitySqAvg = utilitySq;
    parentUtilityStdev = sqrt(
      std::max(
        0.0,
        ((utilitySq + variancePrior) * variancePriorWeight + utilitySqAvg * weightSum)
        / (variancePriorWeight + weightSum - 1.0)
        - utilitySq
      )
    );
  }
  parentUtilityStdevFactor = 1.0 + searchParams.cpuctUtilityStdevScale * (parentUtilityStdev / searchParams.cpuctUtilityStdevPrior - 1.0);

  double parentUtilityForFPU = parentUtility;
  if(searchParams.fpuParentWeightByVisitedPolicy) {
    double avgWeight = std::min(1.0, pow(policyProbMassVisited, searchParams.fpuParentWeightByVisitedPolicyPow));
    parentUtilityForFPU = avgWeight * parentUtility + (1.0 - avgWeight) * getUtilityFromNN(*(node.getNNOutput()));
  }
  else if(searchParams.fpuParentWeight > 0.0) {
    parentUtilityForFPU = searchParams.fpuParentWeight * getUtilityFromNN(*(node.getNNOutput())) + (1.0 - searchParams.fpuParentWeight) * parentUtility;
  }

  double fpuValue;
  {
    double fpuReductionMax = isRoot ? searchParams.rootFpuReductionMax : searchParams.fpuReductionMax;
    double fpuLossProp = isRoot ? searchParams.rootFpuLossProp : searchParams.fpuLossProp;
    double utilityRadius = searchParams.winLossUtilityFactor + searchParams.staticScoreUtilityFactor + searchParams.dynamicScoreUtilityFactor;

    double reduction = fpuReductionMax * sqrt(policyProbMassVisited);
    fpuValue = pla == P_WHITE ? parentUtilityForFPU - reduction : parentUtilityForFPU + reduction;
    double lossValue = pla == P_WHITE ? -utilityRadius : utilityRadius;
    fpuValue = fpuValue + (lossValue - fpuValue) * fpuLossProp;
  }

  return fpuValue;
}


void Search::selectBestChildToDescend(
  SearchThread& thread, SearchNode& node, int nodeState,
  int& numChildrenFound, int& bestChildIdx, Loc& bestChildMoveLoc,
  bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE],
  bool isRoot)
{
  assert(thread.pla == node.nextPla);

  if (using_bts) {
    return selectBestChildToDescendBts(
      thread, node, nodeState, numChildrenFound, bestChildIdx, bestChildMoveLoc, posesWithChildBuf, isRoot);
  }

  double maxSelectionValue = POLICY_ILLEGAL_SELECTION_VALUE;
  bestChildIdx = -1;
  bestChildMoveLoc = Board::NULL_LOC;

  int childrenCapacity;
  const SearchChildPointer* children = node.getChildren(nodeState,childrenCapacity);

  double policyProbMassVisited = 0.0;
  double maxChildWeight = 0.0;
  double totalChildWeight = 0.0;
  const NNOutput* nnOutput = node.getNNOutput();
  assert(nnOutput != NULL);
  const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    Loc moveLoc = children[i].getMoveLocRelaxed();
    int movePos = getPos(moveLoc);
    float nnPolicyProb = policyProbs[movePos];
    policyProbMassVisited += nnPolicyProb;

    int64_t edgeVisits = children[i].getEdgeVisits();
    double childWeight = child->stats.getChildWeight(edgeVisits);

    totalChildWeight += childWeight;
    if(childWeight > maxChildWeight)
      maxChildWeight = childWeight;
  }
  //Probability mass should not sum to more than 1, giving a generous allowance
  //for floating point error.
  assert(policyProbMassVisited <= 1.0001);

  //First play urgency
  double parentUtility;
  double parentWeightPerVisit;
  double parentUtilityStdevFactor;
  double fpuValue = getFpuValueForChildrenAssumeVisited(
    node, thread.pla, isRoot, policyProbMassVisited,
    parentUtility, parentWeightPerVisit, parentUtilityStdevFactor
  );

  std::fill(posesWithChildBuf,posesWithChildBuf+NNPos::MAX_NN_POLICY_SIZE,false);
  bool antiMirror = searchParams.antiMirror && mirroringPla != C_EMPTY && isMirroringSinceSearchStart(thread.history,0);

  //Try all existing children
  //Also count how many children we actually find
  numChildrenFound = 0;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    numChildrenFound++;
    int64_t childEdgeVisits = children[i].getEdgeVisits();

    Loc moveLoc = children[i].getMoveLocRelaxed();
    bool isDuringSearch = true;
    double selectionValue = getExploreSelectionValueOfChild(
      node,policyProbs,child,
      moveLoc,
      totalChildWeight,childEdgeVisits,fpuValue,
      parentUtility,parentWeightPerVisit,parentUtilityStdevFactor,
      isDuringSearch,antiMirror,maxChildWeight,&thread
    );
    if(selectionValue > maxSelectionValue) {
      // if(child->state.load(std::memory_order_seq_cst) == SearchNode::STATE_EVALUATING) {
      //   selectionValue -= EVALUATING_SELECTION_VALUE_PENALTY;
      //   if(isRoot && child->prevMoveLoc == Location::ofString("K4",thread.board)) {
      //     out << "ouch" << "\n";
      //   }
      // }
      maxSelectionValue = selectionValue;
      bestChildIdx = i;
      bestChildMoveLoc = moveLoc;
    }

    posesWithChildBuf[getPos(moveLoc)] = true;
  }

  const std::vector<int>& avoidMoveUntilByLoc = thread.pla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;

  //Try the new child with the best policy value
  Loc bestNewMoveLoc = Board::NULL_LOC;
  float bestNewNNPolicyProb = -1.0f;
  for(int movePos = 0; movePos<policySize; movePos++) {
    bool alreadyTried = posesWithChildBuf[movePos];
    if(alreadyTried)
      continue;

    Loc moveLoc = NNPos::posToLoc(movePos,thread.board.x_size,thread.board.y_size,nnXLen,nnYLen);
    if(moveLoc == Board::NULL_LOC)
      continue;

    //Special logic for the root
    if(isRoot) {
      assert(thread.board.pos_hash == rootBoard.pos_hash);
      assert(thread.pla == rootPla);
      if(!isAllowedRootMove(moveLoc))
        continue;
    }
    if(avoidMoveUntilByLoc.size() > 0) {
      assert(avoidMoveUntilByLoc.size() >= Board::MAX_ARR_SIZE);
      int untilDepth = avoidMoveUntilByLoc[moveLoc];
      if(thread.history.moveHistory.size() - rootHistory.moveHistory.size() < untilDepth)
        continue;
    }

    float nnPolicyProb = policyProbs[movePos];
    if(antiMirror) {
      maybeApplyAntiMirrorPolicy(nnPolicyProb, moveLoc, policyProbs, node.nextPla, &thread);
    }

    if(nnPolicyProb > bestNewNNPolicyProb) {
      bestNewNNPolicyProb = nnPolicyProb;
      bestNewMoveLoc = moveLoc;
    }
  }
  if(bestNewMoveLoc != Board::NULL_LOC) {
    double selectionValue = getNewExploreSelectionValue(
      node,bestNewNNPolicyProb,totalChildWeight,fpuValue,
      parentWeightPerVisit,parentUtilityStdevFactor,
      maxChildWeight,&thread
    );
    if(selectionValue > maxSelectionValue) {
      maxSelectionValue = selectionValue;
      bestChildIdx = numChildrenFound;
      bestChildMoveLoc = bestNewMoveLoc;
    }
  }
}


void Search::selectBestChildToDescendBts(
  SearchThread& thread, SearchNode& node, int nodeState,
  int& numChildrenFound, int& bestChildIdx, Loc& bestChildMoveLoc,
  bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE],
  bool isRoot)
{
  int64_t node_visits = node.stats.visits.load(std::memory_order_acquire);
  int64_t num_legal_moves = 1;
  node.bts_distr_lock.lock();
  bool distributions_created = (node.bts_distr != nullptr);
  if (distributions_created) {
    num_legal_moves = node.valid_moves->size();
  }
  node.bts_distr_lock.unlock();

  // O(A) recompute distributions
  int recompute_freq = num_legal_moves;
  if (!distributions_created || (node_visits % num_legal_moves) == 0) {
    recomputeNodeDistributions(
      thread, node, nodeState, 
      posesWithChildBuf, isRoot, node_visits);
  }

  // O(1) sample, and fill out the values needed by the search
  // hardcoded params, dont want to bother editing search params
  double denominator = 1.0 / std::log(CONST_E + node_visits);
  double lambda_tilde = 0.75 * denominator;
  double lambda = (isRoot ? 0.5 : 0.1) * denominator;

  double weight_unfrm = lambda;
  double weight_nn = (1.0 - lambda) * lambda_tilde;
  //double weight_bst = (1.0-lambda) * (1.0-lambda_tilde)

  node.bts_distr_lock.lock();
  std::shared_ptr<BtsDiscreteUniformDistribution<Loc>> unfrm_distr = node.unfrm_distr;
  std::shared_ptr<BtsCategoricalDistribution<Loc>> nn_distr = node.nn_distr;
  std::shared_ptr<BtsCategoricalDistribution<Loc>> bts_distr = node.bts_distr;
  node.bts_distr_lock.unlock();

  BtsMixedDistribution<Loc> mixed_distr(unfrm_distr, nn_distr, bts_distr, weight_unfrm, weight_nn);
  bestChildMoveLoc = mixed_distr.sample(thread.int_distr, thread.real_distr, thread.rng_gen);

  // Set the things that are expected to be returned, if making new node then index is at the end of the child array
  // If making a new node, then lock in the Loc's index into children array, even if fail to make it for some reason
  node.bts_search_lock.lock();
  numChildrenFound = node.loc_to_child_idx.size();
  if (node.loc_to_child_idx.find(bestChildMoveLoc) != node.loc_to_child_idx.end()) {
    bestChildIdx = node.loc_to_child_idx[bestChildMoveLoc];
  } else {
    bestChildIdx = numChildrenFound;
    node.loc_to_child_idx[bestChildMoveLoc] = bestChildIdx;
  }
  node.bts_search_lock.unlock();
}


void Search::recomputeNodeDistributions(
  SearchThread& thread, SearchNode& node, int nodeState,
  bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE],
  bool isRoot, int64_t num_visits)
{
  // Check if need to make the static parts of distributions
  node.bts_distr_lock.lock();
  std::shared_ptr<std::vector<Loc>> valid_mvs = node.valid_moves;
  node.bts_distr_lock.unlock();
  if (valid_mvs == nullptr) {
    const NNOutput* nn_output = node.getNNOutput();
    const float* policy_probs = nn_output->getPolicyProbsMaybeNoised();

    valid_mvs = std::make_shared<std::vector<Loc>>();
    valid_mvs->reserve(policySize);
    std::shared_ptr<std::unordered_map<Loc,double>> nn_distr_map = std::make_shared<std::unordered_map<Loc,double>>();
    nn_distr_map->reserve(policySize);

    for(int pos = 0; pos<policySize; pos++) {
      Loc loc = NNPos::posToLoc(pos,thread.board.x_size,thread.board.y_size,nnXLen,nnYLen);
      if (thread.history.isLegal(thread.board,loc,thread.pla)) {
        valid_mvs->push_back(loc);
        nn_distr_map->insert_or_assign(loc, policy_probs[pos]);
      }
    }

    std::shared_ptr<BtsDiscreteUniformDistribution<Loc>> unfrm_distr = std::make_shared<BtsDiscreteUniformDistribution<Loc>>(valid_mvs);
    std::shared_ptr<BtsCategoricalDistribution<Loc>> nn_distr = std::make_shared<BtsCategoricalDistribution<Loc>>(nn_distr_map);

    node.bts_distr_lock.lock();
    node.unfrm_distr = unfrm_distr;
    node.nn_distr = nn_distr;
    node.valid_moves = valid_mvs;
    node.bts_distr_lock.unlock();
  }

  // Now below is pretty much just a c&p of computing PUCT weights (Q-vals), and then shoving them into a map
  double max_weight = std::numeric_limits<double>::lowest();
  std::shared_ptr<std::unordered_map<Loc,double>> bts_distr_map = std::make_shared<std::unordered_map<Loc,double>>();
  bts_distr_map->reserve(policySize);

  int childrenCapacity;
  const SearchChildPointer* children = node.getChildren(nodeState,childrenCapacity);

  double policyProbMassVisited = 0.0;
  double maxChildWeight = 0.0;
  double totalChildWeight = 0.0;
  const NNOutput* nnOutput = node.getNNOutput();
  assert(nnOutput != NULL);
  const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    Loc moveLoc = children[i].getMoveLocRelaxed();
    int movePos = getPos(moveLoc);
    float nnPolicyProb = policyProbs[movePos];
    policyProbMassVisited += nnPolicyProb;

    int64_t edgeVisits = children[i].getEdgeVisits();
    double childWeight = child->stats.getChildWeight(edgeVisits);

    totalChildWeight += childWeight;
    if(childWeight > maxChildWeight)
      maxChildWeight = childWeight;
  }
  
  assert(policyProbMassVisited <= 1.0001);

  double parentUtility;
  double parentWeightPerVisit;
  double parentUtilityStdevFactor;
  double fpuValue = getFpuValueForChildrenAssumeVisited(
    node, thread.pla, isRoot, policyProbMassVisited,
    parentUtility, parentWeightPerVisit, parentUtilityStdevFactor
  );

  std::fill(posesWithChildBuf,posesWithChildBuf+NNPos::MAX_NN_POLICY_SIZE,false);
  bool antiMirror = searchParams.antiMirror && mirroringPla != C_EMPTY && isMirroringSinceSearchStart(thread.history,0);

  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    int64_t childEdgeVisits = children[i].getEdgeVisits();

    Loc moveLoc = children[i].getMoveLocRelaxed();
    bool isDuringSearch = true;
    double selectionValue = getExploreSelectionValueOfChild(
      node,policyProbs,child,
      moveLoc,
      totalChildWeight,childEdgeVisits,fpuValue,
      parentUtility,parentWeightPerVisit,parentUtilityStdevFactor,
      isDuringSearch,antiMirror,maxChildWeight,&thread
    );
    if(selectionValue > max_weight) {
      max_weight = selectionValue;
    }

    posesWithChildBuf[getPos(moveLoc)] = true;
    bts_distr_map->insert_or_assign(moveLoc, selectionValue);
  }

  const std::vector<int>& avoidMoveUntilByLoc = thread.pla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;

  // for(int movePos = 0; movePos<policySize; movePos++) {
  for(Loc moveLoc : *valid_mvs) {
    int movePos = NNPos::locToPos(moveLoc, thread.board.x_size, nnOutput->nnXLen, nnOutput->nnYLen);
    bool alreadyTried = posesWithChildBuf[movePos];
    if(alreadyTried)
      continue;

    // Loc moveLoc = NNPos::posToLoc(movePos,thread.board.x_size,thread.board.y_size,nnXLen,nnYLen);
    if(moveLoc == Board::NULL_LOC)
      continue;

    // if(isRoot) {
    //   assert(thread.board.pos_hash == rootBoard.pos_hash);
    //   assert(thread.pla == rootPla);
    //   if(!isAllowedRootMove(moveLoc))
    //     continue;
    // }
    // if(avoidMoveUntilByLoc.size() > 0) {
    //   assert(avoidMoveUntilByLoc.size() >= Board::MAX_ARR_SIZE);
    //   int untilDepth = avoidMoveUntilByLoc[moveLoc];
    //   if(thread.history.moveHistory.size() - rootHistory.moveHistory.size() < untilDepth)
    //     continue;
    // }

    float nnPolicyProb = policyProbs[movePos];
    // if(antiMirror) {
    //   maybeApplyAntiMirrorPolicy(nnPolicyProb, moveLoc, policyProbs, node.nextPla, &thread);
    // }

    // Edited here to get new explore selection values for all potential new nodes
    double selection_value = getNewExploreSelectionValue(
      node,nnPolicyProb,totalChildWeight,fpuValue,
      parentWeightPerVisit,parentUtilityStdevFactor,
      maxChildWeight,&thread
    );

    if(selection_value > max_weight) {
      max_weight = selection_value;
    }

    bts_distr_map->insert_or_assign(moveLoc, selection_value);
  }

  // finally make the bts distribution and update it
  double temp = get_bts_temp(num_visits);
  for (std::pair<Loc,double> pr : *bts_distr_map) {
    Loc loc = pr.first;
    double weight = pr.second;
    double bts_weight = std::exp((weight - max_weight) / temp);
    bts_distr_map->insert_or_assign(loc,bts_weight);
  }

  std::shared_ptr<BtsCategoricalDistribution<Loc>> bts_distr = std::make_shared<BtsCategoricalDistribution<Loc>>(bts_distr_map);
  node.bts_distr_lock.lock();
  node.bts_distr = bts_distr;
  node.bts_distr_lock.unlock();
}

double Search::get_bts_temp(int64_t num_visits) {
  // hardcoded init temp, sorry
  double init_temp = 0.5;
  double min_temp = 1.0e-6;
  double temp = init_temp / std::log(CONST_E + num_visits);
  if (temp < min_temp) {
    return min_temp;
  }
  return temp;
}
