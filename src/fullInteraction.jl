function fullInteraction(data, numVars, bernO)
m = bernO+1;
MATRIX = data;
designMATRIX = zeros(size(data,1),1);
if numVars>1
    for j = 1:1:numVars-1
        starti = size(designMATRIX,2);
        for i =1:1:m^j
            designMATRIX = hcat(designMATRIX, MATRIX[:,i].*data[:,j*m+1:(j+1)*m]);
        end
        designMATRIX = designMATRIX[:,2:end]
        if j<numVars-1
            MATRIX=designMATRIX;
            designMATRIX= zeros(size(data,1),1);
        end
    end
else
    designMATRIX = data;
end

return designMATRIX
end
