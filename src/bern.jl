function bern(t, order)
# construct bernstein polynomial of order ``order"
out = zeros(size(t,1),1)
for o = 0:order
   if o ==0
      out = b(t,order,o)
   else
      out = [out b(t,order,o)];
   end

end
return out
end
