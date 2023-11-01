classdef subnetwork_classC
    properties
        num_of_subnetworks int16 = 10
        deploy_length double = 10  
        factoryArea double = [10, 10]
        subnet_radius double = 3
        minD double = 1 
        minDistance double = 1*3
        sigmaS double = 0
        transmit_power_dBm double = 0
        transmit_power double = db2pow(0);
        bandwidth double = 5e6
        frequency double = 6e9
        lambdA double = 3e8/6e9
        plExponent double = 2.7
        mapXPoints
        mapYPoints
        powers
        distance
        noisePower
        powerLevels
        correlationDistance double = 2
        numChannel = 1
        type char = 'rand'
        transTime double = 0.05e0
        sampTime double = 0.05e0
        updateTime double = 2
        speed double = 2
        num_of_steps int16 = 10
        count int16 = 1
        num_neighb int16 = 1
        rateTarget = 0
    end
    methods
        function obj = subnetwork_classC(N, S_sd, K, powerL, type)
            if nargin == 1
                obj.num_of_subnetworks = N;
            elseif nargin == 2
                obj.num_of_subnetworks = N;
                obj.sigmaS = S_sd;
            elseif nargin == 3
                obj.num_of_subnetworks = N;
                obj.sigmaS = S_sd;
                obj.numChannel = K;
            else
                obj.num_of_subnetworks = N;
                obj.sigmaS = S_sd;
                obj.numChannel = K;
                obj.type = type;
            end
            obj.mapXPoints = linspace(0,obj.deploy_length,601);
            obj.mapYPoints =  linspace(0,obj.deploy_length,601);
            obj.num_of_steps = round(obj.updateTime/obj.sampTime);
            obj.noisePower = db2pow(-114);
            obj.powerLevels = powerL;
        end
        function [mapp] = createMap(obj, Gamma)
            N1 = length(obj.mapXPoints);
            N2 = length(obj.mapYPoints);
            Z = randn(N1,N2) + 1j*randn(N1,N2);
            mapp = real(fft2((Gamma.*Z)/sqrt(N1*N2)))*sqrt(2);
        end
        function [Gamma] = computeGamma(obj)
            N1 = length(obj.mapXPoints);
            N2 = length(obj.mapYPoints);
            G = zeros(N1,N2);
            for n=1:N1 
                for m=1:N2
                     G(n,m)=exp(-1*sqrt(min(abs(obj.mapXPoints(1)-obj.mapXPoints(n)), ...
                     max(obj.mapXPoints)-abs(obj.mapXPoints(1)-obj.mapXPoints(n)))^2 + min(abs(obj.mapYPoints(1)-obj.mapYPoints(m)), ...
                     max(obj.mapYPoints)-abs(obj.mapYPoints(1)-obj.mapYPoints(m)))^2)/obj.correlationDistance);  
                   
                end
            end
            Gamma = sqrt(fft2(G));
        end
        function [S, dist] = computeShadowing(obj,Loc,map)
            X = obj.mapXPoints;
            Y = obj.mapYPoints;
            [~,idx] = ismember(round(squeeze(Loc(1,:)),1),round(X,1));
            [~,idy] = ismember(round(squeeze(Loc(2,:)),1),round(Y,1));
            idxx = sub2ind(size(map),idx,idy);
            f = map(idxx);
            fAB = f+f';
            dist = pdist2(Loc',Loc');
            S = ((1-exp(-1*dist/obj.correlationDistance))./(sqrt(2)*sqrt((1+exp(-1*dist/obj.correlationDistance))))).*fAB;
        end
        function loc = Mobility(obj)
            X = zeros(obj.num_of_subnetworks,1);
            Y = zeros(obj.num_of_subnetworks,1);
            gwLoc = zeros(obj.num_of_subnetworks,2);
            XBound1 = obj.factoryArea(1)-2*obj.subnet_radius;
            YBound1 = obj.factoryArea(2)-2*obj.subnet_radius;
            dist_2 = obj.minDistance^2;
            loop_terminate = 1;
            nValid = 1;
            while nValid <= obj.num_of_subnetworks && loop_terminate < 1e6
                newX = XBound1*(rand()-0.5);
                newY = YBound1*(rand()-0.5);
                if min((X(1:nValid) - newX).^2 + (Y(1:nValid) - newY).^2)>dist_2
                    X(nValid) = newX;
                    Y(nValid) = newY;
                    nValid = nValid+1;
                end
                loop_terminate = loop_terminate+1;
            end
            if nValid < obj.num_of_subnetworks
                loc = -1;
            else
                
                Xtemp = X+obj.factoryArea(1)/2;
                Ytemp = Y +obj.factoryArea(2)/2;
                gwLoc(:,1) = Xtemp;
                gwLoc(:,2) = Ytemp;
                Xway = obj.factoryArea(1)/2;
                Yway = obj.factoryArea(2)/2;
                N = round(obj.updateTime/obj.sampTime);
                D = reshape(atan2(Yway-Ytemp,Xway-Xtemp),[],1);
                loc = zeros(2,obj.num_of_subnetworks,N);
                loc(1,:,1) = Xtemp;
                loc(2,:,1) = Ytemp;
                Imat = zeros(obj.num_of_subnetworks,obj.num_of_subnetworks);
                Imat = Imat+diag(190*ones(obj.num_of_subnetworks,1));
                for n=2:N
                    Xtemp = Xtemp+cos(D).*obj.speed*obj.transTime;
                    Ytemp = Ytemp+ sin(D).*obj.speed*obj.transTime;
                    loc_temp = [Xtemp,Ytemp];
                    dist_pw = pdist2(loc_temp,loc_temp);
                    dist_pw = dist_pw+Imat;
                    [indx1,~] = find(dist_pw <= obj.minDistance);
                    indx11 = unique(indx1);
                    D(indx11) = D(indx11)+pi; 
                    Xtemp(indx11) = Xtemp(indx11)+cos(D(indx11))*obj.speed*obj.sampTime;
                    Ytemp(indx11) = Ytemp(indx11)+sin(D(indx11))*obj.speed*obj.sampTime;
                    Xtemp(find(Xtemp < obj.subnet_radius)) = obj.subnet_radius;
                    Xtemp(find(Xtemp > XBound1)) = XBound1;
                    Ytemp(find(Ytemp < obj.subnet_radius)) = obj.subnet_radius;
                    Ytemp(Ytemp > XBound1) = XBound1;
                    D(find(Xtemp==obj.subnet_radius)) = rand(length(Xtemp(find(Xtemp==obj.subnet_radius))),1)*2*pi;
                    D(find(Xtemp==XBound1)) = rand(length(Xtemp(find(Xtemp==XBound1))),1)*2*pi;
                    D(find(Ytemp==obj.subnet_radius)) = rand(length(Ytemp(find(Ytemp==obj.subnet_radius))),1)*2*pi;
                    D(find(Ytemp==YBound1)) = rand(length(Ytemp(find(Ytemp==YBound1))),1)*2*pi;
                    loc(1,:,n) = Xtemp;
                    loc(2,:,n) = Ytemp;
                end
            end
        end
        function [powerr] = compute_power(obj, dist, S)
            N = obj.num_of_subnetworks;
            S_linear = db2pow(obj.sigmaS*S);
            h = (1/sqrt(2))*(randn(N,N)+1j*randn(N,N));
            powerr = obj.transmit_power*(4*pi/obj.lambdA)^(-2)*(min(1,dist.^(-1*obj.plExponent))).*S_linear.*(abs(h)^2);  
        end
        function [powers, distance] = generate_samples1(obj)
            N = obj.num_of_subnetworks;
            distance = zeros(obj.num_of_steps,N,N);
            powers = zeros(obj.num_of_steps,N,N);
            [gwLoc] = Mobility(obj);

            dist_rand = rand(obj.num_of_subnetworks,obj.num_of_steps)*(obj.subnet_radius - obj.minD) + obj.minD;
            angN = rand(obj.num_of_subnetworks,obj.num_of_steps)*2*pi;
            D_XLoc = squeeze(gwLoc(1,:,:)) + dist_rand.*cos(angN);
            D_YLoc = squeeze(gwLoc(2,:,:)) + dist_rand.*sin(angN);
            dvLoc(1,:,:)= D_XLoc;
            dvLoc(2,:,:) = D_YLoc;
            %allLoc = [gwLoc; dvLoc];
            
%             dist = pdist2(gwLoc, dvLoc);
%             param.gwLoc = gwLoc;

            if strcmp(obj.type, 'corr')
                Gamma = obj.computeGamma();
            end
            for k = 1:obj.num_of_steps  
                if strcmp(obj.type, 'corr')
                    mapp = obj.createMap(Gamma);
                end
                %[allloc, dist, obj] = create_layout(obj);
                dist = pdist2(squeeze(gwLoc(:,:,k)).',squeeze(dvLoc(:,:,k)).');
          
                distg = pdist2(squeeze(gwLoc(:,:,k)).',squeeze(gwLoc(:,:,k)).');
                if strcmp(obj.type, 'corr')
                    allLoc = [squeeze(gwLoc(:,:,k)), squeeze(dvLoc(:,:,k))];
                    
                    [S, dist] = obj.computeShadowing(allLoc, mapp);
                    %size(S)
                    %dist(1:N, N+1:end)
                    pw = compute_power(obj, dist(1:N, N+1:end),S(1:N, N+1:end));
                    powers(k,:,:) = pw;
                else
                    S = 1*randn(N,N);
                    powers(k,:,:) = obj.compute_power(dist,S);
                end
                distance(k,:,:) = distg;
            end
        end
        function [sir, sinr, intPower, rate] = calculate_rate(obj, action)
            if action == 0
                action = randi(obj.numChannel, obj.num_of_subnetworks, 1);
            end
            all_powers = squeeze(obj.powers(obj.count,:,:));
            mask = eye(obj.num_of_subnetworks);
            receivePower = sum(all_powers.*mask);
            interP = all_powers.*(1-mask);
            for m = 1:obj.numChannel
                indm = action == m;
                Itemp =  interP*indm;
                intPower(m,:) = Itemp+ obj.noisePower;
                sir(m,:) = receivePower(:)./Itemp; 
            end
            for n = 1:obj.num_of_subnetworks
                sinr(n) = receivePower(n)/(intPower(action(n),n));
            end
            rate = log2(1+sinr); 

        end
        function [state, obj] = reset(obj)
            obj.count = 1;
            [obj.powers, obj.distance] = obj.generate_samples1();
            %action = obj.numChannel*ones(obj.num_of_subnetworks,1);
            %powerVector = obj.powerLevels(action);
            %%
            all_powers = squeeze(obj.powers(obj.count,:,:));
            mask = eye(obj.num_of_subnetworks);
            receivePower = sum(all_powers.*mask);
            interP = all_powers.*(1-mask);
            intPower = sum(interP,2);
            %reward = rate;
            %reward(1:end) = mean(rate(:));
            %reward(1:end) = mean(rate);
            state_ind = (pow2db([receivePower(:).'; intPower(:).'])+44)/(6.2);

            
            distt = squeeze(obj.distance(obj.count,:,:));
            state = [];
            for ii = 1:obj.num_of_subnetworks
                 [~,Indx] = sort(distt(ii,:), 'ascend');
                 state = [state reshape(state_ind(:,Indx(1:obj.num_neighb)),[],1)];
            end
        end
        function [state, obj] = resetC(obj)
            obj.count = 1;
            [obj.powers, obj.distance] = obj.generate_samples1();
            [~, ~, intPower, ~] = obj.calculate_rate(0);
            state_ind = (pow2db(intPower)+114)/(94);
            distt = squeeze(obj.distance(obj.count,:,:));
            state = [];
            for ii = 1:obj.num_of_subnetworks
                 [~,Indx] = sort(distt(ii,:), 'ascend');
                 state = [state reshape(state_ind(:,Indx(1:obj.num_neighb)),[],1)];
            end
        end
        function [next_state, reward ,isdone, obj] = stepC(obj,action)
            %Calculate SINR, rate, and interference power
            %process to get next_state
            obj.count = obj.count + 1;
            [~, ~, intPower, rate] = obj.calculate_rate(action);
            isdone = 0;
            reward = rate;
            reward(reward < obj.rateTarget) = reward(reward < obj.rateTarget) -0*(obj.rateTarget - reward(reward < obj.rateTarget));
            %reward(1:end) = min(rate(:));
            state_ind = (pow2db(intPower)+114)/(94);
            distt = squeeze(obj.distance(obj.count,:,:));
            next_state = [];
            for ii = 1:obj.num_of_subnetworks
                 [~,Indx] = sort(distt(ii,:), 'ascend');
                 next_state = [next_state reshape(state_ind(:,Indx(1:obj.num_neighb)),[],1)];
            end
        end
        function [next_state, reward ,isdone, obj] = step(obj,action)
            %Calculate SINR, rate, and interference power
            %process to get next_state
            obj.count = obj.count + 1;
            [sir, ~, ~, rate] = obj.calculate_rateP(action);
            isdone = 0;
            powerVector = obj.powerLevels(action);
            %%
            all_powers = squeeze(obj.powers(obj.count,:,:));
            mask = eye(obj.num_of_subnetworks);
            receivePower = sum(all_powers.*mask);
            interP = all_powers.*(1-mask);
            intPower = sum(interP,2);
           
            reward = rate;
            %reward(1:end) = mean(rate(:));
            %reward(1:end) = mean(rate);
            state_ind = (pow2db([receivePower(:).'; intPower(:).'])+114)/(94); %[powerVector(:).'; logsir(:).'];
            %state_ind = (pow2db(squeeze(obj.powers(obj.count,:,:)))+121.0150)/(93.5891);
            distt = squeeze(obj.distance(obj.count,:,:));
            next_state = [];
            for ii = 1:obj.num_of_subnetworks
                 [~,Indx] = sort(distt(ii,:), 'ascend');
                 next_state = [next_state reshape(state_ind(:,Indx(1:obj.num_neighb)),[],1)];
            end
        end
    end
end
