function main_beam_deflection_newton_raphson()
    % =====================================================================
    % 1. PHYSICAL PARAMETERS (Exactly the same)
    % =====================================================================
    L  = 0.5;          % Length of the beam (meters)
    b  = 0.03;         % Width of cross-section (meters)
    t  = 0.01;         % Thickness of cross-section (meters)
    P  = 1000.0;       % Point load at free tip (N)
    w_area = 38500;    % Distributed load per area (N/m^2)
    E  = 200e9;        % Young's Modulus: 200 GPa
    I  = 2.50e-09;     % Moment of Inertia (m^4)
    
    q = w_area * b;    % Convert area load to linear load
    EI = E * I;        % Flexural rigidity
    
    % =====================================================================
    % 2. MESH DISCRETIZATION
    % =====================================================================
    num_steps = 150; 
    s_mesh = linspace(0, L, num_steps);
    
    % =====================================================================
    % 3. SYSTEM OF FIRST-ORDER ODEs (Anonymous Function)
    % =====================================================================
    f_system = @(s, Y) [Y(2); ...
                        Y(3); ...
                        -(q/EI)*(L - s) - (P/EI)];
                    
    % =====================================================================
    % 4. THE SHOOTING METHOD (Using NEWTON-RAPHSON Iteration)
    % =====================================================================
    % Newton-Raphson requires only ONE starting guess
    current_guess = 0.0; 
    
    tolerance = 1e-9;
    max_iter = 50;
    converged = false;
    
    % Perturbation step size for calculating the numerical derivative
    dg = 1e-7; 
    
    fprintf('Starting Newton-Raphson Root Search...\n');
    fprintf('%-10s %-20s %-20s\n', 'Iter', 'Current Guess', 'Tip Moment Error');
    
    for iter = 1:max_iter
        % Step 4.1: Find the baseline error at the current guess
        sol_current = RK4(f_system, s_mesh, [0; 0; current_guess]);
        error_current = sol_current(3, end); % Target target boundary condition is 0
        
        fprintf('%-10d %-20.6f %-20.6e\n', iter, current_guess, error_current);
        
        % Step 4.2: Check for convergence
        if abs(error_current) < tolerance
            fprintf('Convergence achieved in %d iterations!\n', iter);
            final_solution = sol_current;
            converged = true;
            break;
        end
        
        % Step 4.3: Perturb the guess slightly to calculate the derivative
        sol_perturbed = RK4(f_system, s_mesh, [0; 0; current_guess + dg]);
        error_perturbed = sol_perturbed(3, end);
        
        % Step 4.4: Compute the numerical derivative (dError / dGuess)
        derivative = (error_perturbed - error_current) / dg;
        
        % Avoid division by zero safety constraint
        if abs(derivative) < 1e-12
            error('Derivative close to zero. Newton-Raphson method failed.');
        end
        
        % Step 4.5: NEWTON-RAPHSON UPDATE FORMULA
        % x_next = x_current - f(x) / f'(x)
        current_guess = current_guess - (error_current / derivative);
    end
    
    if ~converged
        error('The Newton-Raphson shooting method failed to converge.');
    end
    
    % =====================================================================
    % 5. EXTRACT ANGLE & CALCULATE X-DIRECTION DEFLECTION
    % =====================================================================
    vertical_y  = final_solution(1, :);   
    slope_angle = final_solution(2, :);   
    
    % True deformed horizontal coordinates
    deformed_x = cumtrapz(s_mesh, cos(slope_angle));
    
    % Horizontal displacement profile (u_x = s - x)
    horiz_deflection_x = s_mesh - deformed_x;
    
    % Print out results
    fprintf('\n--- RESULTS AT THE FREE TIP (s = L) ---\n');
    fprintf('Deflection Angle:          %.4f degrees\n', rad2deg(slope_angle(end)));
    fprintf('Vertical Deflection (y):    %.4f mm\n', vertical_y(end) * 1000);
    fprintf('Horizontal Deflection (x):  %.4f mm\n', horiz_deflection_x(end) * 1000);
    
    % =====================================================================
    % 6. PLOT THE TRUE 2D DEFORMED SHAPE
    % =====================================================================
    figure;
    plot(deformed_x, vertical_y, 'b-', 'LineWidth', 2.5);
    grid on;
    title('Deformed Cantilever Profile (Newton-Raphson Solver)');
    xlabel('Horizontal Position x (m)');
    ylabel('Vertical Deflection y (m)');
    set(gca, 'YDir', 'reverse'); 
end

% =========================================================================
% 7. RUNGE-KUTTA 4 SYSTEM SOLVER (Exactly your original implementation)
% =========================================================================
function y = RK4(f, x, y0)
    y = zeros(length(y0), length(x)); 
    y(:, 1) = y0; 
    h = x(2) - x(1); 
    n = length(x);
    for i = 1:n-1
        k1 = f(x(i), y(:, i));
        k2 = f(x(i) + h/2, y(:, i) + h*k1/2);
        k3 = f(x(i) + h/2, y(:, i) + h*k2/2);
        k4 = f(x(i) + h, y(:, i) + h*k3);
        y(:, i+1) = y(:, i) + h*(k1 + 2*k2 + 2*k3 + k4)/6;
    end
end