function myBuffer = storeExperienceM(myBuffer,observation,action,nextObservation,reward,isDone,N)
        % storeExperience stores an experience in a replay buffer.

     
    
        %myBuffer.bufferIndex = myBuffer.bufferIndex + 1;
        if myBuffer.bufferIndex > myBuffer.bufferSize
            myBuffer.bufferIndex = 1;
            myBuffer.currentBufferLength = myBuffer.bufferSize;
        end
        myBuffer.currentBufferLength = max(myBuffer.bufferIndex+N,myBuffer.currentBufferLength);
        n = myBuffer.bufferIndex+1;
        m = myBuffer.bufferIndex+N;
        myBuffer.observation(:,n:m) = observation;
        myBuffer.action(:,n:m) = action;
        myBuffer.nextObservation(:,n:m) = nextObservation;
        myBuffer.reward(:,n:m) = reward;
        myBuffer.isDone(:,n:m) = isDone;
        myBuffer.bufferIndex = myBuffer.bufferIndex + N;
end