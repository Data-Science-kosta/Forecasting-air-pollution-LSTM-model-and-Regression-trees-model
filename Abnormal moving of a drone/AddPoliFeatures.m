function Xout = AddPoliFeatures(X)
[m, n] = size(X);
Xout = [X ones(m, 1)];
for i = 1:n
   for j = 1:n
        Xout = [Xout (Xout(:,i).^2).*(Xout(:,j).^2)]; 
   end
end

end