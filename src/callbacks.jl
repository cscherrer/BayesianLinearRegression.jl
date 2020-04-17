export stopAtIteration

function stopAtIteration(n)
    function(m)
        if m.iterationCount > n
            m.done = true
        end
    end
end

export stopAfter

function stopAfter(Δt)
    stopTime = time() + Δt
    function(m)
        if time() > stopTime
            m.done = true
        end
    end
end

export fixedEvidence

function fixedEvidence()
    logEv0 = -Inf
    function(m)
        logEv = logEvidence(m)
        
        if logEv == logEv0 
            m.done = true
        end
        logEv0 = logEv
        return logEv
    end
end
