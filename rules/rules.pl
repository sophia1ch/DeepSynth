:- use_module(library(apply)).

%%% Facts %%%
orientation(vertical).
orientation(flat).
orientation(upright).
orientation(upside_down).
orientation(cheesecake).
%orientation(weird).

color(blue).
color(yellow).
color(red).

shape(pyramid).
shape(wedge).
shape(block).

% Grounded = specific case: handled like attribute and not an interaction, so check with attribute rules !!!
interaction(grounded). % Never the only attribute in a rule, so always in combination with an orientation, color or shape attribute!
interaction(touching(_)).
interaction(pointing(_)).
interaction(on_top_of(_)).
%interaction(inside(_)).

max_items(5).
min_items(1).


%%% Generating %%%
% Generate repeatedly until a structure satisfies all checks (main function)
generate_valid_structure(Checks, Structure) :-
    determine_structure_size(Checks, N),
    repeat,
    generate_items(N, N, Structure),
    interaction_constraint_check(Structure),
    (and(Checks) -> !; fail).

% Generating with a limit, if the given query is to complicated (probability of generating it, is to low) then stop after a specific number of attempts
% Not so easy to implement (currently not complete), because the recursive function doesn`t work with generating new structures every new attempt.
% That is because the function writes Structure in the first loop and in the later loop it is already written, what concludes in errors.
% Changing the output to a new `Variable` doesn`t work either, because Structure is the variable which is tested in `Checks`
% Currently this error is captured by the generating_scene function and threading
%generate_valid_structure(Checks, Structure) :-
%    generate_valid_structure_limit(Checks, Structure, 1000).
%
%generate_valid_structure_limit(_, _, 0) :-
%    write("End"), nl,
%    !,
%    fail.
%
%generate_valid_structure_limit(Checks, Structure, Attempts) :-
%    Attempts > 0,
%    NextAttempts is Attempts - 1,
%    write("tried"), nl,
%    generate_structure(Structure),
%    write(Attempts), write("generated: "), write(Structure), nl,
%    (interaction_constraint_check(Structure) ->
%        (and(Checks) -> !; generate_valid_structure_limit(Checks, Structure, NextAttempts));
%        generate_valid_structure_limit(Checks, Structure, NextAttempts)
%    ).


% Generate repeatedly until a structure doesnt fulfills the checks
generate_invalid_structure(Checks, Structure) :-
    determine_invalid_structure_size(Checks, N),
    repeat,
    generate_items(N, N, Structure),
    interaction_constraint_check(Structure),
    (not(and(Checks)) -> !; fail).

% Define preferred size range if not specified by rules
default_size_range([7, 6, 5, 4, 3, 2, 1]).

determine_structure_size(Checks, N) :-
    possible_structure_sizes_from_checks(Checks, ValidSizes),
    random_member(N, ValidSizes).

contains_explicitly_allows_1_in_either_or(Checks) :-
    member(C, Checks),
    (
        C = either_or(1, _, _)
    ;
        C = either_or(_, 1, _)
    ).

possible_structure_sizes_from_checks(Checks, ValidSizes) :-
    default_size_range(DefaultSizes),

    % 1. Check for either_or constraints
    ( member(either_or(A, B, _), Checks) ->
        AllowedSizes1 = [A, B]
    ; AllowedSizes1 = DefaultSizes ),

    % 2. Check for odd/even number constraints
    ( member(odd_number_of(_), Checks) ->
        include(odd, AllowedSizes1, AllowedSizes2)
    ; member(even_number_of(_), Checks) ->
        include(even, AllowedSizes1, AllowedSizes2)
    ; AllowedSizes2 = AllowedSizes1 ),

    % 3. Find all minimums from any constraints that imply minimum structure size
    findall(N, (
        member(C, Checks),
        (
            C = exactly(_, N, _)
        ;   C = exactly(_, _, N, _)
        ;   C = at_least_interaction(_, _, _, N, _)
        )
    ), Ns),
    ( Ns = [] -> MinReq = 1 ; max_list(Ns, MinReq) ),

    % 4. Filter by minimum requirement
    include(ge(MinReq), AllowedSizes2, ValidSizes),
    ValidSizes \= [], !.  % Succeed only if non-empty

determine_invalid_structure_size(Checks, N) :-
    invalid_structure_sizes_from_checks(Checks, InvalidSizes),
    random_member(N, InvalidSizes).

invalid_structure_sizes_from_checks(Checks, Sizes) :-
    default_size_range(DefaultSizes),

    % Step 1: Exclude sizes listed in either_or
    ( member(either_or(A, B, _), Checks) ->
        exclude(eq(A), DefaultSizes, T1),
        exclude(eq(B), T1, Sizes1)
    ; Sizes1 = DefaultSizes ),

    % Step 2: Apply opposite of odd/even
    ( member(odd_number_of(_), Checks) ->
        include(even, Sizes1, Sizes2)
    ; member(even_number_of(_), Checks) ->
        include(odd, Sizes1, Sizes2)
    ; Sizes2 = Sizes1 ),

    % Step 3: Filter out any structure sizes that exactly match attribute-based counts
    findall(N, (
        member(C, Checks),
        (C = exactly(_, N, _); C = exactly(_, _, N, _))
    ), ExactNs),
    exclude_list(Sizes2, ExactNs, Sizes),

    Sizes \= [], !.  % Fail safely if nothing matches


generate_structure(Structure) :-
    random_structure_size(N),
    repeat,
    (generate_items(N, N, Structure) -> !; fail).

generate_items(0, _, []) :- !.
generate_items(N, TotalN, [item(Id,C,S,O,I)|Rest]) :-
    Id is N - 1,
    random_color(C),
    random_shape(S),
    random_orientation(S, O),
    MaxId is TotalN - 1,
    random_interaction(MaxId, Id, I),
    N1 is N - 1,
    generate_items(N1, TotalN, Rest).

random_color(Color) :-
    findall(C, color(C), Colors),
    random_member(Color, Colors).

random_shape(Shape) :-
    findall(S, shape(S), Shapes),
    random_member(Shape, Shapes).

random_orientation(Shape, Orientation) :-
    %findall(O, orientation(O), Orientations),
    % exclude specific cases for the different shapes
    findall(O, (orientation(O), (
        (Shape = block, O \= cheesecake);
        (Shape = pyramid, O \= cheesecake);
        (Shape = wedge, true)
    )), ValidOrientations),
    random_member(Orientation, ValidOrientations).

random_interaction(MaxId, Id, Interaction) :-
    findall(I, interaction(I), Interactions),
    random_member(Template, Interactions),
    assign_random_id(Template, MaxId, Id, Interaction).

assign_random_id(grounded, _, _, grounded).
assign_random_id(touching(_), MaxId, Id, touching(T)) :-
    random_between(0, MaxId, T),
    T \= Id.
assign_random_id(pointing(_), MaxId, Id, pointing(T)) :-
    random_between(0, MaxId, T),
    T \= Id.
assign_random_id(on_top_of(_), MaxId, Id, on_top_of(T)) :-
    random_between(0, MaxId, T),
    T \= Id.
assign_random_id(inside(_), MaxId, Id, inside(T)) :-
    random_between(0, MaxId, T),
    T \= Id.

item_has_attribute(Attr, item(_,C,S,O,I)) :-
    Attr = C; Attr = S; Attr = O; (nonvar(I), I =.. [Attr|_]).

item_has_two_attributes(A1, A2, Item) :-
    A2 \= grounded,
    item_has_attribute(A1, Item),
    item_has_attribute(A2, Item).

item_has_two_attributes(A1, grounded, Item) :-
     % Check if one of the attributes is grounded, if so we have a special case
     item_has_attribute(A1, Item),
     (item_has_attribute(grounded, Item);
     item_has_attribute(pointing, Item);
     item_has_attribute(touching, Item)).

count_attribute(Attr, Structure, Count) :-
    include(item_has_attribute(Attr), Structure, Filtered),
    length(Filtered, Count).

count_multiple_attributes(A1, A2, Structure, Count) :-
    include(item_has_two_attributes(A1, A2), Structure, Filtered),
    length(Filtered, Count).

% Counts how many unique items have QAttr and are in an interaction with a target with IAttr.
count_interaction_matching_items(QAttr, IAttr, touching, Structure, Count) :-
    findall(SourceId,
        (
            member(item(SourceId, SC, SS, SO, touching(TargetId)), Structure),
            item_has_attribute(QAttr, item(SourceId, SC, SS, SO, touching(TargetId))),
            member(item(TargetId, TC, TS, TO, _), Structure),
            item_has_attribute(IAttr, item(TargetId, TC, TS, TO, _))
        ),
        Matches1),
    findall(SourceId,
        (
            member(item(TargetId, TC, TS, TO, touching(SourceId)), Structure),
            item_has_attribute(IAttr, item(TargetId, TC, TS, TO, touching(SourceId))),
            member(item(SourceId, SC, SS, SO, _), Structure),
            item_has_attribute(QAttr, item(SourceId, SC, SS, SO, _))
        ),
        Matches2),
    append(Matches1, Matches2, All),
    sort(All, Unique),
    length(Unique, Count).

count_interaction_matching_items(QAttr, IAttr, InteractionName, Structure, Count) :-
    InteractionName \= touching,
    findall(SourceId,
        (
            member(item(SourceId, SC, SS, SO, Interaction), Structure),
            Interaction =.. [InteractionName, TargetId],
            item_has_attribute(QAttr, item(SourceId, SC, SS, SO, Interaction)),
            member(item(TargetId, TC, TS, TO, _), Structure),
            item_has_attribute(IAttr, item(TargetId, TC, TS, TO, _))
        ),
        Matches),
    sort(Matches, Unique),
    length(Unique, Count).

% Check if an item has QAttr and an interaction of type InteractionName that leads to another item with IAttr
has_interaction_attribute(QAttr, IAttr, InteractionName, Structure, item(_,C,S,O,I)) :-
    item_has_attribute(QAttr, item(_,C,S,O,I)),
    decode_interaction(I, InteractionName, Target),
    Target \= none, % Not really needed, because if grounded, we don`t use this case
    member(item(Target,TC,TS,TO,TI), Structure),
    item_has_attribute(IAttr, item(Target,TC,TS,TO,TI)).

% Decode the interaction to see if it matches the given name and extract the target if any.
decode_interaction(grounded, grounded, none).
decode_interaction(grounded, Name, none) :- Name \= grounded.
decode_interaction(I, Name, T) :-
    I =.. [Name,T], !.
decode_interaction(_, _, none).

% Check different constraints for the interaction between items
interaction_constraint_check(Structure) :-
    %% On_top_of %%
    % Ensure all on_top_of attributes have the right orientation and check if the stacking is valid
    forall(
        member(item(SourceId,_,SourceShape,SourceOrientation,on_top_of(TargetId)), Structure),
        (
            member(item(TargetId, _, TargetShape, TargetOrientation, TargetInteraction), Structure),
            (
                % Grounded cases
                (TargetInteraction = grounded,
                 ((TargetShape = block);
                  (TargetShape = wedge, TargetOrientation = cheesecake);
                  (TargetShape = pyramid, SourceShape = pyramid, TargetOrientation = SourceOrientation, TargetOrientation = vertical);
                  (TargetShape = pyramid, SourceShape = pyramid, TargetOrientation = SourceOrientation, TargetOrientation = upright)));
                % Pointing cases
                (TargetInteraction = pointing(SourceId),
                 ((TargetShape = block);
                  (TargetShape = wedge, TargetOrientation = cheesecake);
                  (TargetShape = pyramid, SourceShape = pyramid, TargetOrientation = SourceOrientation, TargetOrientation = vertical);
                  (TargetShape = pyramid, SourceShape = pyramid, TargetOrientation = SourceOrientation, TargetOrientation = upright)));
                % Touching cases
                (TargetInteraction = touching(_),
                 ((TargetShape = block);
                  (TargetShape = wedge, TargetOrientation = cheesecake);
                  (TargetShape = pyramid, SourceShape = pyramid, TargetOrientation = SourceOrientation, TargetOrientation = vertical);
                  (TargetShape = pyramid, SourceShape = pyramid, TargetOrientation = SourceOrientation, TargetOrientation = upright)));
                % On top of another item, source is on top of target
                (TargetInteraction = on_top_of(AnotherId),
                 AnotherId \= SourceId,
                 ((TargetShape = block);
                  (TargetShape = wedge, TargetOrientation = cheesecake);
                  (TargetShape = pyramid, TargetOrientation = vertical, SourceOrientation = upright);
                  (TargetShape = pyramid, TargetOrientation = upright, TargetOrientation = SourceOrientation);
                  % Upside-down pyramid stacking inside another upside-down pyramid
                  (TargetShape = pyramid, SourceShape = pyramid,
                   TargetOrientation = upside_down, SourceOrientation = upside_down);

                  % Upside-down pyramid on block
                  (TargetShape = block, SourceShape = pyramid, SourceOrientation = upside_down, TargetOrientation = SourceOrientation)))
            )
        )
    ),
    % Allow a flat block to have up to two items on_top_of it; other items can only have one
    % Currently the rule is turned off: If you want to activate, change the Count of the rule
    forall(
        member(item(TargetId, _, TargetShape, TargetOrientation, _), Structure),
        (
            (TargetShape = block, TargetOrientation = flat ->
                (
                    findall(SourceId, member(item(SourceId, _, _, _, on_top_of(TargetId)), Structure), Sources),
                    length(Sources, Count),
                    % Change to 2, if you want to activate the two objects on top of a flat block rule
                    Count =< 2
                );
                (
                    findall(SourceId, member(item(SourceId, _, _, _, on_top_of(TargetId)), Structure), Sources),
                    length(Sources, Count),
                    Count =< 1
                )
            )
        )
    ),
    % Ensure no loops in the on_top_of chain
    forall(
        member(item(SourceId, _, _, _, on_top_of(_)), Structure),
        \+ has_loop(SourceId, Structure)
    ),
    %% Touching %%
    % Every item can only have max 4 touching items (above = on_top_of, below = none)
    \+ (
        member(item(TargetId,_,_,_,_), Structure),
        findall(SourceId,
                (member(item(SourceId,_,_,_,Interaction), Structure),
                Interaction = touching(TargetId)),
                Sources),
        length(Sources, Count),
        Count > 4
    ),
    % Upside-down pyramids must either be touched by at least 2 others or be on top of an upside-down block
    forall(
        member(item(PyramidId, _, pyramid, upside_down, Interaction), Structure),
        (
            % Option 1: touched by at least 2 valid pieces
            findall(OtherId,
                (
                    member(item(OtherId, _, OtherShape, OtherOrientation, touching(PyramidId)), Structure),
                    (
                        OtherShape = block
                        ;
                        (OtherShape \= block, OtherOrientation = upright)
                    )
                ),
                ValidTouching),
            length(ValidTouching, Count),
            Count >= 2
            ;
            % Option 2: on_top_of an upside-down pyramid stack, base is upside-down block
            Interaction = on_top_of(ParentId),
            upside_down_stack_has_block_base(ParentId, Structure)
        )
    ),
    % Upside-down wedges must be touched by at least 2 other pieces
    forall(
        member(item(WedgeId, _, wedge, upside_down, _), Structure),
        (
            findall(OtherId,
                (
                    member(item(OtherId, _, OtherShape, OtherOrientation, touching(WedgeId)), Structure),
                    (
                        OtherShape = block
                        ;
                        (OtherShape = pyramid, OtherOrientation = upright)
                    )
                ),
                ValidTouching
            ),
            length(ValidTouching, Count),
            Count >= 2
        )
    ),
    % Ensure at least one touching element is grounded via grounded or pointing (all pointing are grounded)
    % Or the touching item is part of a chain of touching items
    forall(
        member(item(SourceId,_,_,_,touching(TargetId)), Structure),
        (member(item(TargetId,_,_,_,grounded), Structure);
        member(item(TargetId,_,_,_,touching(_)), Structure);
        member(item(TargetId,_,_,_,pointing(_)), Structure))
    ),
    %% Pointing %%
    % All pointing items are grounded! Loops allowed
    % Flat and cheesecake items can point only to grounded items
    % Upright and vertical items can only point to items with on_top_of(SourceId)
    forall(
        member(item(SourceId, _, _, _, pointing(TargetId)), Structure),
        ((member(item(SourceId, _, _, flat, _), Structure),
        member(item(TargetId, _, _, _, grounded), Structure);
        member(item(SourceId, _, _, cheesecake, _), Structure),
        member(item(TargetId, _, _, _, grounded), Structure));
        (member(item(SourceId, _, _, upright, _), Structure),
        member(item(TargetId, _, _, _, on_top_of(SourceId)), Structure);
        member(item(SourceId, _, _, vertical, _), Structure),
        member(item(TargetId, _, _, _, on_top_of(SourceId)), Structure)))
    ).

% Helper functions for checking, if on_top_of creates a loop!
has_loop(StartId, Structure) :-
    has_loop_helper(StartId, StartId, Structure, []).

% If the current ID has already been visited, there`s a loop
has_loop_helper(_, CurrentId, _, Visited) :-
    member(CurrentId, Visited),
    !.
has_loop_helper(StartId, CurrentId, Structure, Visited) :-
    member(item(CurrentId, _, _, _, on_top_of(NextId)), Structure),
    has_loop_helper(StartId, NextId, Structure, [CurrentId|Visited]).

% Recursively walk the on_top_of chain to find the base of an upside-down pyramid stack.
upside_down_stack_has_block_base(ItemId, Structure) :-
    member(item(ItemId, _, block, upside_down, _), Structure).

upside_down_stack_has_block_base(ItemId, Structure) :-
    member(item(ItemId, _, pyramid, upside_down, on_top_of(ParentId)), Structure),
    upside_down_stack_has_block_base(ParentId, Structure).

%%% Rules %%%
% Check predicates (pure checks, no generation)
at_least(Attr, N, Structure) :-
    count_attribute(Attr, Structure, C),
    C >= N.

at_least(A1, A2, N, Structure) :-
    count_multiple_attributes(A1, A2, Structure, C),
    C >= N.

% Generic version for any interaction that involves a target item
at_least_interaction(QAttr, IAttr, InteractionName, N, Structure) :-
    findall(Item,
        (member(Item, Structure),
        has_interaction_attribute(QAttr, IAttr, InteractionName, Structure, Item)),
        Filtered),
    length(Filtered, Count),
    Count >= N.

exactly(Attr, N, Structure) :-
    count_attribute(Attr, Structure, C),
    C =:= N.

exactly(A1, A2, N, Structure) :-
    count_multiple_attributes(A1, A2, Structure, C),
    C =:= N.

% Generic version for any interaction that involves a target item
exactly_interaction(QAttr, IAttr, InteractionName, N, Structure) :-
    findall(Item,
        (member(Item, Structure),
        has_interaction_attribute(QAttr, IAttr, InteractionName, Structure, Item)),
        Filtered),
    length(Filtered, Count),
    Count =:= N.

more_than(A1, A2, Structure) :-
    count_attribute(A1, Structure, C1),
    count_attribute(A2, Structure, C2),
    C1 > C2.

odd_number_of(Structure) :-
    length(Structure, L),
    1 is L mod 2.

even_number_of(Structure) :-
    length(Structure, L),
    L \= 0,
    0 is L mod 2.

odd_number_of(Attr, Structure) :-
    count_attribute(Attr, Structure, C),
    1 is C mod 2.

odd_number_of(A1, A2, Structure) :-
    count_multiple_attributes(A1, A2, Structure, C),
    1 is C mod 2.

odd_number_of_interaction(QAttr, IAttr, InteractionName, Structure) :-
    count_interaction_matching_items(QAttr, IAttr, InteractionName, Structure, C),
    1 is C mod 2.

even_number_of(Attr, Structure) :-
    count_attribute(Attr, Structure, C),
    C \= 0,
    0 is C mod 2.

even_number_of(A1, A2, Structure) :-
    count_multiple_attributes(A1, A2, Structure, C),
    C \= 0,
    0 is C mod 2.

even_number_of_interaction(QAttr, IAttr, InteractionName, Structure) :-
    count_interaction_matching_items(QAttr, IAttr, InteractionName, Structure, C),
    C \= 0,
    0 is C mod 2.

either_or(N1, N2, Structure) :-
    length(Structure, L),
    (L =:= N1; L =:= N2).

and([]).
and([C|Cs]) :-
    call(C),
    and(Cs).

or([C|_]) :-
    call(C).
or([_|Cs]) :-
    or(Cs).

odd(N) :- 1 is N mod 2.
even(N) :- 0 is N mod 2.
ge(Min, N) :- N >= Min.
eq(X, Y) :- X =:= Y.

exclude_list(List, Excluded, Result) :-
    exclude(in_list(Excluded), List, Result).

in_list(L, X) :- member(X, L).