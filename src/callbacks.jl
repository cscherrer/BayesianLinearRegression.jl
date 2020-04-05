
function stopAtIteration(n)
    function(m)
        if m.iterationCount > n
            m.done = true
        end
    end
end

function stopAfter(Δt)
    stopTime = time() + Δt
    function(m)
        if time() > stopTime
            m.done = true
        end
    end
end
