function [model] = verifySigma(model)
% Verify the sigma values of the membership functions
% If a sigma value is negative, set it to 0.1.

for i = 1:length(model.Inputs)

    for j = 1:length(model.Inputs(1,i).MembershipFunctions)
    
        if model.Inputs(1,i).MembershipFunctions(1,j).Parameters(1) <= 0
            model.Inputs(1,i).MembershipFunctions(1,j).Parameters(1) = 0.1;
        end
        if model.Inputs(1,i).MembershipFunctions(1,j).Parameters(2) <= 0
            model.Inputs(1,i).MembershipFunctions(1,j).Parameters(2) = 0.1;
        end
    end
end

for i = 1:length(model.Outputs.MembershipFunctions)

    for j = 1:1:length(model.Outputs.MembershipFunctions(1,i).Parameters)-1

        if model.Outputs.MembershipFunctions(1,i).Parameters(j) <= 0
            model.Outputs.MembershipFunctions(1,i).Parameters(j) = 0.1;
        end
    end
end

end
