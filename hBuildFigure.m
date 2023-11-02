function [trainingPlot,lineReward,lineAveReward,ax] = hBuildFigure()

    % Copyright 2021 The MathWorks, Inc.

    plotRatio = 16/9;
    trainingPlot = figure(...
        'Visible','off',...
        'HandleVisibility','off', ...
        'NumberTitle','off',...
        'Name','Cart Pole Custom Training (DQN agent)');
    trainingPlot.Position(3) = plotRatio * trainingPlot.Position(4);

    ax = gca(trainingPlot);

    lineReward = animatedline(ax);
    lineAveReward = animatedline(ax,'Color','r','LineWidth',3);
    xlabel(ax,'Episode')
    ylabel(ax,'Reward')
    legend(ax,'Cumulative Reward','Average Reward','Location','northwest')
    title(ax,'Training Progress')
end