function dbern(t, order)
out = [];
for o = 0:1:order
   if o ==0
      out = db(t,order,o)
   else
      out = [out db(t,order,o)];
   end

end
return out
end
