
% Plotting Stablity/Torque data for Journal Bearing Problem
% 


%We = [0, 0.1, 0.25, 0.5, 1.0];

lambda = [0, 0.1, 0.25];
chiMa001 = [0.48882, 0.7289 ,0.9578];
chiMa01 = [0.49882, 0.7419, 0.97661];
chiMa05 = [0.50113, 0.7623, 1.0223];

torMa001 = [0, 0 ,0];
torMa01 = [0, 0, 0];
torMa05 = [0, 0, 0];



figure(1)
plot(lambda, chiMa001,'-*','linewidth', 1.5)
xlabel('$\lambda_D$', 'interpreter', 'latex'); ylabel('\chi')
hold on
plot(lambda, chiMa01,'-*','linewidth', 1.5)
hold on
plot(lambda, chiMa05,'-*','linewidth', 1.5)
hold on 
legend('Ma=0.001','Ma=0.01','Ma=0.05')
saveas(figure(1),'journal_stability_fene.png')
close(figure(1))


figure(2)
plot(We, torMa001,'-*','linewidth', 1.5)
xlabel('$\lambda_D$', 'interpreter', 'latex'); ylabel('C')
hold on
plot(We, torMa01,'-*','linewidth', 1.5)
hold on
plot(We, torMa05,'-*','linewidth', 1.5)
hold on 
legend('Ma=0.001','Ma=0.01','Ma=0.05')
saveas(figure(2),'journal_torque.png')
close(figure(2))
