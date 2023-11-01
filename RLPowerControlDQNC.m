%% Distributed Channel Allocation for Mobile 6G Subnetworks via Multi-Agent Deep Q-Learning
clc
clear all
N = 25;                     %
S_sd = 5; 
rngSeed = 1;
rng(rngSeed);
type = 'corr';
powerLevels = db2pow(0); 
K = 4;
obj = subnetwork_classC(N, S_sd, K, powerLevels, type);

%%
rng(0)
state_size = K*obj.num_neighb;
obsInfo = rlNumericSpec([state_size 1]); 
 
actInfo = rlFiniteSetSpec([1:K+1]);

numObservations = obsInfo.Dimension(1);
numActions = numel(actInfo.Elements);

qNetwork = [
    featureInputLayer(obsInfo.Dimension(1),'Name','state')
    fullyConnectedLayer(24,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    %lstmLayer(64,'Name','lstm1')
    fullyConnectedLayer(24, 'Name','CriticStateFC2')
    reluLayer('Name','CriticCommonRelu')
%     fullyConnectedLayer(24, 'Name','CriticStateFC3')
%     reluLayer('Name','CriticCommonRelu1')
    fullyConnectedLayer(length(actInfo.Elements),'Name','output')];
qNetwork = dlnetwork(qNetwork);

critic = rlVectorQValueFunction(qNetwork,obsInfo,actInfo);

optimizerOpt = rlOptimizerOptions(...
    LearnRate=1e-3, ...
    GradientThreshold=1);
criticOptimizer = rlOptimizer(optimizerOpt);

policy = rlMaxQPolicy(critic);
%% Create buffers
myBuffer.bufferSize = 1e5;
myBuffer.bufferIndex = 0;
myBuffer.currentBufferLength = 0;
myBuffer.observation = zeros(numObservations,myBuffer.bufferSize);
myBuffer.nextObservation = ...
                       zeros(numObservations,myBuffer.bufferSize);
myBuffer.action = ...
    zeros(1,myBuffer.bufferSize);
myBuffer.reward = zeros(1,myBuffer.bufferSize);
myBuffer.isDone = zeros(1,myBuffer.bufferSize);

numEpisodes = 2000;
maxStepsPerEpisode = 40-1;
discountFactor = 0.99;
aveWindowSize = 10;
trainingTerminationValue = 220;

warmStartSamples = 2000;
numEpochs = 1;
miniBatchSize = 256;
horizonLength = 2;

epsilon = 1;
epsilonMin = 0.001;
epsilonDecay = 0.0001;
targetUpdateFrequency = 4;
realRatio = 1; % Set to 1 to run a standard DQN
numGradientSteps = 5; %was 5
episodeCumulativeRewardVector = [];
%end
[trainingPlot,lineReward,lineAveReward,ax] = hBuildFigure;
modelType = 'centralized';
%%
% Train the policy for the maximum number of episodes or until the average
% reward indicates that the policy is sufficiently trained.
totalStepCt = 0;
targetCritic = critic;
modelTrainedAtleastOnce = false;
allR = [];
maxDelay = 1;
switchIndicator = 1;
for episodeCt = 1:numEpisodes
    sIndx = randi(maxDelay, obj.num_of_subnetworks,1);
    Action = randi(K, 1, obj.num_of_subnetworks);
    Action_temp = Action;
    % 1. Reset the environment at the start of the episode
    %obs = reset(env)
    [state, obj] = obj.resetC();
    ep_reward = 0;
    %for n = 1:N
    episodeReward = zeros(maxStepsPerEpisode,1);

    % 2. Create buffers to store experiences. The dimensions for each buffer
    % must be as follows.
    %
    % For observation buffer: 
    %     numberOfObservations x numberOfObservationChannels x batchSize
    %
    % For action buffer: 
    %     numberOfActions x numberOfActionChannels x batchSize
    %
    % For reward buffer: 
    %     1 x batchSize
    %
%     observationBuffer = zeros(numObs,1,maxStepsPerEpisode*N);
%     actionBuffer = zeros(numAct,1,maxStepsPerEpisode*N);
%     rewardBuffer = zeros(1,maxStepsPerEpisode*N);
    %end
    % 3. Generate experiences for the maximum number of steps per
    % episode or until a terminal condition is reached.
    for stepCt = 1:maxStepsPerEpisode
        totalStepCt = totalStepCt +1;
        % Compute an action using the policy based on the current 
        % observation.
        
        for n = 1:N
            if sIndx(n) == switchIndicator
                if rand() < epsilon
                    action = actInfo.usample;
                else
                    action = getAction(policy,{state(:,n)});
                end
                Action(n) = action{1};
                if action{1} ~= K+1
                    Action_temp(n) = action{1};
                end
            end
        end

        if totalStepCt > warmStartSamples
                epsilon = max(epsilon*(1-epsilonDecay),...
                              epsilonMin);
        end
        % Apply the action to the environment and obtain the resulting
        % observation and reward.
        switchedC = find(sIndx == switchIndicator);
        Nswitch = numel(switchedC);
        [next_state, reward ,isDone, obj] = obj.stepC(Action_temp.');
        % Store the action, observation, and reward experiences in buffers.
        myBuffer = storeExperienceM(myBuffer,...
                                state(:,switchedC),...
                                Action(:,switchedC),...
                                next_state(:,switchedC),reward(switchedC),isDone,Nswitch);

        episodeReward(stepCt) = mean(reward);

        
        state = next_state;
        ep_reward = ep_reward + mean(reward);

        switchIndicator = switchIndicator+1;
        if switchIndicator > maxDelay
            switchIndicator = 1;
        end
         % Train DQN agent
        for gradientCt = 1:numGradientSteps
            if myBuffer.currentBufferLength >= miniBatchSize ...
                    && totalStepCt>warmStartSamples
                %----------------------------------------------
                % 4. Sample minibatch from experience buffers.
                %----------------------------------------------
                [sampledObservation,...
                    sampledAction,...
                    sampledNextObservation,...
                    sampledReward,...
                    sampledIsdone] ...
                          = sampleMinibatch(...
                                modelTrainedAtleastOnce,...
                                realRatio,...
                                miniBatchSize,...
                                myBuffer,[]);

                %----------------------------------------------
                % 5. Compute target Q value.
                %----------------------------------------------
                % Compute target Q value
                [targetQValues, MaxActionIndices] = ...
                        getMaxQValue(targetCritic, ...
                        {reshape(sampledNextObservation,...
                        [numObservations,1,miniBatchSize])});

                % Compute target for nonterminal states
                targetQValues(~logical(sampledIsdone)) = ... 
                    sampledReward(~logical(sampledIsdone)) + ...
                    discountFactor.*...
                    targetQValues(~logical(sampledIsdone));
                % Compute target for terminal states
                targetQValues(logical(sampledIsdone)) = ...
                    sampledReward(logical(sampledIsdone));

                lossData.batchSize = miniBatchSize;
                lossData.actInfo = actInfo;
                lossData.actionBatch = sampledAction;
                lossData.targetQValues = targetQValues;

                %----------------------------------------------
                % 6. Compute gradients.
                %----------------------------------------------
                criticGradient = ...
                    gradient(critic,...
                         @criticLossFunction, ...
                        {reshape(sampledObservation,...
                        [numObservations,1,miniBatchSize])},...
                        lossData);

                %----------------------------------------------
                % 7. Update the critic network using gradients.
                %----------------------------------------------
                [critic, criticOptimizer] = update(...
                    criticOptimizer, critic,...
                    criticGradient);

                % Update the policy parameters using the critic
                % parameters.
                policy = setLearnableParameters(...
                                policy,...
                                getLearnableParameters(critic));
            end
        end
        % Update target critic periodically
        if mod(totalStepCt, targetUpdateFrequency)==0
            targetCritic = critic;
        end
        
        
    end
    allR = [allR ep_reward/maxStepsPerEpisode];
    %---------------------------------------------------------
    % 8. Update the training visualization.
    %---------------------------------------------------------
    episodeCumulativeReward = sum(episodeReward);
    episodeCumulativeRewardVector = cat(2,...
        episodeCumulativeRewardVector,episodeCumulativeReward);
    movingAveReward = movmean(episodeCumulativeRewardVector,...
        aveWindowSize,2);
    addpoints(lineReward,episodeCt,episodeCumulativeReward);
    addpoints(lineAveReward,episodeCt,movingAveReward(end));
    title(ax, "Training Progress - Episode: " + episodeCt + ...
        ", Total Step: " + string(totalStepCt) + ...
        ", epsilon:" + string(epsilon))
    drawnow;

    if episodeCt > 2
        if allR(episodeCt) >= max(allR(1:episodeCt-1))
            save('Models/ChannelAllocationplusnoswitch', 'policy')
        end
    end
end
save('DataChannelAllocationplusnoswitch','allR')
%%
clc
clear all
N = 45;
S_sd = 5; 
rngSeed = 1;
rng(rngSeed);
type = 'corr';
powerLevels = db2pow(0); 
K = 4;
obj = subnetwork_classC(N, S_sd, K, powerLevels, type);

rng default % For reproducibility
maxDelay = 10;
switchIndicator = 1;

 load('C:\Users\ra\MATLAB Drive\GNN\Models\ChannelAllocationD21nNeigh.mat')
 maxStepsPerEpisode = 40-1;

 intcon = 1:N;

s= 1/1;

lb = [ones(1,N)]/s;
ub = [K*ones(1,N)]/s;
opts = optimoptions('surrogateopt','PlotFcn',[],"ConstraintTolerance",1e-6);

opts.MaxFunctionEvaluations = 400;
rng default % For reproducibility


A = [];
b = [];
Aeq = [];
beq = [];
for episodeCt = 1:1
    sIndx = randi(maxDelay, obj.num_of_subnetworks,1);
    Action = randi(K, 1, obj.num_of_subnetworks);
    [state,obj] = obj.resetC();
    for m = 1:maxStepsPerEpisode
        error = db2pow(0*randn(N,N));
        pow = abs(pow2db(squeeze(obj.powers(obj.count,:,:))).*(1-eye(N)));
        tic
        out_cgc = centralizedColoring(pow+pow2db(error), K, 'Greedy');
        time_cgc(episodeCt,m) = toc;
        cgc_allocation(episodeCt,m,:) = out_cgc; 
        tic
        out_rdm = randi(K, N, 1);
        time_rdm(episodeCt,m) = toc;
        time_dqn(episodeCt,m) = 0;
        for n = 1:N
            if sIndx(n) == switchIndicator
                tic
                action = getAction(policy,{state(:,n)});
                st = toc
                time_dqn(episodeCt,m) = time_dqn(episodeCt,m) + st;
                Action(n) = action{1};
            end
        end
        
        H=squeeze(obj.powers(obj.count,:,:));
        f = @(P)objectiveFunc(P,H,N,s);
        tic
        [sol,fval,eflag,outpt] = surrogateopt(f,lb,ub,intcon,opts);
        time_iter(episodeCt,m) = toc;
%         iter_allocation(episodeCt,m,:) = sol;
        dqn_allocation(episodeCt,m,:) = Action;
        %==end of iterative algorithm
        out_seq = Action;
%         for n = 1:N
%             Indx_cgc = find(out_cgc == out_cgc(n));
%             Indx_rdm = find(out_rdm == out_rdm(n)); 
%             Indx_seq = find(out_seq == out_seq(n));
%             %Indx_iter = find(sol == sol(n));
%             cap_cgc(episodeCt,m,n) = log2(1+H(n,n)/(sum(H(n, Indx_cgc(Indx_cgc ~= n)))+ db2pow(-114)));
%             cap_rdm(episodeCt,m,n) = log2(1+H(n,n)/(sum(H(n, Indx_rdm(Indx_rdm ~= n)))+ db2pow(-114)));
%             cap_seq(episodeCt,m,n) = log2(1+H(n,n)/(sum(H(n, Indx_seq(Indx_seq ~= n)))+ db2pow(-114)));
%            % cap_iter(episodeCt,m,n) = log2(1+H(n,n)/(sum(H(n, Indx_iter(Indx_iter ~= n)))+ db2pow(-114)));
%         end
        [next_state, reward ,isDone, obj] = obj.stepC(Action.');
        state = next_state;

        switchIndicator = switchIndicator+1;
        if switchIndicator > maxDelay
            switchIndicator = 1;
        end
    end
    episodeCt
end
%save('ChannelAllocation1TestResuts4LargeNet','cap_cgc','cap_seq','cap_rdm','time_cgc','time_dqn','time_rdm','dqn_allocation','cgc_allocation')
%%
load('DataChannelAllocationD21nNeigh.mat')
allR1 = allR;
load('DataChannelAllocationD24nNeigh.mat')
allR4 = allR;
load('DataChannelAllocationD28Neigh.mat')
allR8 = allR;
tau1 = 20;
tau2= 300;
figure(); hold on
%plot([1 2000], [5.0745 5.0745], '-m', 'DisplayName','Surrogate Optimizer')
plot(movmean(allR1,tau1),'--b', 'linewidth',1,'DisplayName', 'Reward: $|\mathcal{D}| = 0$')
plot(movmean(allR1,tau2),'-b','DisplayName', 'Averaged Reward: $|\mathcal{D}| = 0$')

plot(movmean(allR4,tau1),'--r', 'linewidth',1,'DisplayName', 'Reward: $|\mathcal{D}| = 3$')
plot(movmean(allR4,tau2),'-r','DisplayName', 'Averaged Reward: $|\mathcal{D}| = 3$')
plot(movmean(allR8,tau1),'--k', 'linewidth',1,'DisplayName', 'Reward: $|\mathcal{D}| = 7$')
plot(movmean(allR8,tau2),'-k','DisplayName', 'Averaged Reward: $|\mathcal{D}| = 7$')
%plot([1 2000], [3.9308 3.9308], '-g', 'DisplayName','Random')
legend show
xlim([0 2000])
xlabel('Episode')
ylabel('Reward [bps/Hz]')
grid on
%%
expName = 'sumRate';
savefig(tt,['LearningCurve',expName])
exportgraphics(tt,['LearningCurve',expName,'.pdf'])
%%
load('ChannelAllocation1TestResutsConst0Neighb.mat')
capacity_cgc = cap_cgc;
capacity_seq1 = cap_seq;
capacity_rdm = cap_rdm;
load('ChannelAllocation1TestResuts4nNeighb.mat')
capacity_seq4 = cap_seq;
% load('ChannelAllocation1TestResuts8Neighb.mat')
% capacity_seq8 = cap_seq;
%load('ChannelAllocation1TestResuts1.mat')
load('ChannelAllocation1TestResuts1nNeighb.mat')
capacity_seq4 = cap_seq;
%%
[x1,c1] = ecdf(capacity_cgc(:));
[x2,c2] = ecdf(capacity_rdm(:));
[x3,c3] = ecdf(capacity_seq1(:));
[x4,c4] = ecdf(capacity_seq4(:));
%[x5,c5] = ecdf(capacity_seq8(:));
%[x6,c6] = ecdf(cap_iter(:));
figure(); hold on
%plot(c6,x6,'--b' ,'DisplayName','Surogate Optimizer')
plot(c1,x1,'-b' ,'DisplayName','Centralized Coloring')
plot(c3,x3, '-r', 'DisplayName','MADDQN: : $|\mathcal{D}| = 0$')
plot(c4,x4, '--r', 'DisplayName','MADDQN: : $|\mathcal{D}| = 3$')
%plot(c5,x5, '-.r', 'DisplayName','MADDQN: : $|\mathcal{D}| = 7$')
plot(c2,x2, '-k', 'DisplayName','Random')
xlim([0 10])
xlabel('Per device rate [bps/Hz]')
ylabel('CDF')
legend show
legend('location','northwest')
grid on
%set(gca,'YScale','log')
%xlim([0.001 0.5])
%set(gca,'XScale','log')
%title('25 cells - correlated shadowing')

%%
C1=(min(capacity_seq1(:,10:end,:),[],3));
C4=(min(capacity_seq4(:,10:end,:),[],3));
Cc=(min(capacity_cgc,[],3));
Cr=(min(capacity_rdm,[],3))
figure; cdfplot(C1(:)); hold on; cdfplot(C4(:)); 
%%
expName = 'Sumrate'
savefig(tt,['IndividualRateCDF',expName])
exportgraphics(tt,['IndividualRateCDF',expName,'.pdf'])
%%
%subplot(2,1,2)
scgc = sum(capacity_cgc,3)/N;
srdm = sum(capacity_rdm,3)/N;
sseq1 = sum(capacity_seq1,3)/N;
sseq4 = sum(capacity_seq4,3)/N;
%sseq8 = sum(capacity_seq8,3)/N;
siter = sum(cap_iter,3)/N;


[x1,c1] = ecdf(scgc(:));
[x2,c2] = ecdf(srdm(:));
[x3,c3] = ecdf(sseq1(:));
[x4,c4] = ecdf(sseq4(:));
%[x5,c5] = ecdf(sseq8(:));
[x6,c6] = ecdf(siter(:));
figure(); hold on
plot(c6,x6,'--b' ,'DisplayName','Surogate Optimizer')
plot(c1,x1,'-b' ,'DisplayName','Centralized Coloring')
plot(c3,x3, '-r', 'DisplayName','MADDQN: : $|\mathcal{D}| = 0$')
plot(c4,x4, '--r', 'DisplayName','MADDQN: : $|\mathcal{D}| = 3$')
%plot(c5,x5, '-.r', 'DisplayName','MADDQN: : $|\mathcal{D}| = 7$')
plot(c2,x2, '-k', 'DisplayName','Random')
xlim([2 6])
xlabel('Average rate [bps/Hz]')
ylabel('CDF')
legend show
legend('location','northwest')
grid on
%%
expName = 'Sumrate'
savefig(tt,['MeanRateCDF',expName])
exportgraphics(tt,['MeanRateCDF',expName,'.pdf'])
%%

intcon = 1:N;

s= 1/1;

lb = [ones(1,N)]/s;
ub = [K*ones(1,N)]/s;

opts = optimoptions('surrogateopt',"ConstraintTolerance",1e-6);
opts.MaxFunctionEvaluations = 400;
rng default % For reproducibility


A = [];
b = [];
Aeq = [];
beq = [];

 

for kk = 1:1
   [~,obj] = obj.reset();
for k =4:4 %maxStepsPerEpisode
    H=squeeze(obj.powers(obj.count,:,:));
    f = @(P)objectiveFunc(P,H,N,s);
    [sol,fval,eflag,outpt] = surrogateopt(f,lb,ub,intcon,opts);
end
kk
end
%%
function f = objectiveFunc(P,H,N,s)
P = round(P*s);
all_powers = H; 
mask = eye(N);
receivePower = sum(all_powers.*mask);
interP = all_powers.*(1-mask);
f = 0;
f1 = [];
for n = 1:N
    Indx = find(P == P(n));
    f = f+log2(1+ receivePower(n)/(sum(interP(n,Indx))+db2pow(-114)));
    %f1 = min([f1, log2(1+ receivePower(n)/(sum(interP(n,Indx))+db2pow(-114)))]);
end
f = -1*f/N; %-0.5*f1;
end
